@echo off
set srcfile=D:\aurora_data\file_paths_aurora.txt
set dest=D:\aurora_data\initial_supine_data

rem Check if destination folder exists, if not create it
if not exist "%dest%" mkdir "%dest%"

rem Loop through each line in the srcfile
for /f "usebackq tokens=*" %%i in ("%srcfile%") do (
    rem Copy each file to the destination folder
    copy "%%i" "%dest%"
)

echo Files copied successfully!
pause
