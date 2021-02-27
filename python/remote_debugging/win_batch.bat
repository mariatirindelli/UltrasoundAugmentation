#!binsh
# This file is called ~script.sh

cd C:\GitRepo\pyfl\remote_debugging

mkdir C:\GitRepo\tmp
mkdir C:\GitRepo\tmp\pyfl

robocopy C:\GitRepo\pyfl C:\GitRepo\tmp\pyfl /MIR /XD C:\GitRepo\pyfl\.git C:\GitRepo\pyfl\.idea C:\GitRepo\pyfl\.plyaxon *__pycache__
scp -r C:\GitRepo\tmp\pyfl mariatirindelli@10.23.0.56:/mnt/data/mariatirindelli

ssh mariatirindelli@10.23.0.56

dos2unix /mnt/data/mariatirindelli/pyfl/remote_debugging/script.sh