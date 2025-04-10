import os
import winshell
from win32com.client import Dispatch
import sys

def create_shortcut(batch_path, icon_path=None):
    # Get desktop path
    desktop = winshell.desktop()
    
    # Create shortcut path
    shortcut_path = os.path.join(desktop, "Sangeet.lnk")
    
    # Create shortcut
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(shortcut_path)
    
    # Set properties
    shortcut.Targetpath = batch_path
    shortcut.WorkingDirectory = os.path.dirname(batch_path)
    
    # Set icon if provided
    if icon_path and os.path.exists(icon_path):
        shortcut.IconLocation = icon_path
    
    # Save shortcut
    shortcut.save()
    
    return shortcut_path

def main(exe_path , ico_path):
    with open(os.path.join(os.getcwd() , "sangeet.bat") , "w") as ms:
        data = f'''
@echo off
:: Replace with your sangeet dir where cloned
cd "{os.getcwd()}" 
python start_server.py
pause
'''
        ms.write(data)
    try:
        shortcut_path = create_shortcut(exe_path, ico_path)
        print(f"Shortcut created successfully at: {shortcut_path}")
    except Exception as e:
        print(f"Error creating shortcut: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
