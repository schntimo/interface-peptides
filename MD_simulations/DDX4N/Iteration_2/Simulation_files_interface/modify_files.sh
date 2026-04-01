find . -type f -name "job_scr*" | while read file; do
    dos2unix "$file"
done