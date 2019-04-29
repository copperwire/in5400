rm -f IN5400_assignment1.zip 
zip -r IN5400_assignment1.zip . -x   "*.idea*" "*.ipynb_checkpoints*" "*data/*" "*storedModels*"  "*README.txt" "*collectSubmission.sh"
