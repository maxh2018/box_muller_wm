rm -rf mywork/.git
git rm --cached ZoDiac
#上传大文件失败
git filter-branch -f --index-filter 'git rm --cached --ignore-unmatch system_status/test_26457'


