#!/usr/bin/env bash
# 隐私自证脚本：敏感串 / 绝对路径 / 私密文件三项都必须 0 命中。
# 提交前跑一遍，CI 里也跑同样的检查。
#
# 用法：bash scripts/privacy_check.sh
set -uo pipefail

cd "$(git rev-parse --show-toplevel)"
FAILED=0

report() {
  local name="$1"; shift
  local out
  out=$("$@" 2>/dev/null)
  if [ -n "$out" ]; then
    echo "❌ $name —— 命中："
    echo "$out" | head -20
    FAILED=1
  else
    echo "✅ $name —— 0 命中"
  fi
}

# ---- 1. 敏感串（数据目录：真实个人信息只可能出现在这里）----
# 手机号模式带前后非数字边界，否则 19 位的腾讯 postId 会被误判
report "data/ 无手机号形态" \
  git grep -n -I -E "(^|[^0-9])1[3-9][0-9]{9}([^0-9]|$)" -- data

# 邮箱排除 RFC 保留的 example.*（文档用的保留域）
EMAIL_OUT=$(git grep -n -I -E "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|net|cn|io|me)([^A-Za-z0-9.-]|$)" -- data 2>/dev/null \
  | grep -v -E "example\.(com|org|net)")
if [ -n "$EMAIL_OUT" ]; then
  echo "❌ data/ 无邮箱形态 —— 命中："; echo "$EMAIL_OUT" | head -20; FAILED=1
else
  echo "✅ data/ 无邮箱形态 —— 0 命中"
fi

report "data/ 无薪资行" \
  git grep -n -I -E "期望薪资\s*[:：]|薪资期望\s*[:：]" -- data

# ---- 2. 代码里不硬编码真实个人信息 ----
# （脱敏脚本里的「薪资」是**脱敏规则的正则定义**，不是数据，不算泄露）
report "代码/测试 无硬编码手机号" \
  git grep -n -I -E "(^|[^0-9])1[3-9][0-9]{9}([^0-9]|$)" -- scripts tests jobfit

EMAIL_OUT=$(git grep -n -I -E "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|org|net|cn|io|me)([^A-Za-z0-9.-]|$)" -- scripts tests jobfit 2>/dev/null \
  | grep -v -E "example\.(com|org|net)")
if [ -n "$EMAIL_OUT" ]; then
  echo "❌ 代码/测试 无硬编码邮箱 —— 命中："; echo "$EMAIL_OUT" | head -20; FAILED=1
else
  echo "✅ 代码/测试 无硬编码邮箱 —— 0 命中"
fi

# ---- 3. 绝对路径（提交物不能带本机路径）----
report "无绝对路径" \
  git grep -n -I -E "[A-Za-z]:\\\\(Users|Program Files|programmer)|/home/[a-z]|/Users/[a-z]" -- . ':!README.md'

# ---- 4. 私密文件未入库 ----
PRIVATE=$(git ls-files | grep -E "作战板|personal-background|\.docx$|\.pdf$" || true)
if [ -n "$PRIVATE" ]; then
  echo "❌ 私密文件未入库 —— 命中："; echo "$PRIVATE"; FAILED=1
else
  echo "✅ 私密文件未入库 —— 0 命中"
fi

echo
if [ "$FAILED" -eq 0 ]; then
  echo "全部通过 ✅"
  exit 0
fi
echo "有未通过项 ❌"
exit 1
