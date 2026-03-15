Set fso = CreateObject("Scripting.FileSystemObject")
Set wmi = GetObject("winmgmts:\\.\root\cimv2")
pidFile = fso.BuildPath(fso.GetParentFolderName(WScript.ScriptFullName), ".litellm.pid")

If Not fso.FileExists(pidFile) Then WScript.Quit

Set f = fso.OpenTextFile(pidFile, 1)
pid = Trim(f.ReadLine)
f.Close

KillTree wmi, CLng(pid)
fso.DeleteFile pidFile

Sub KillTree(wmiObj, parentPid)
    Set children = wmiObj.ExecQuery( _
        "SELECT ProcessId FROM Win32_Process WHERE ParentProcessId = " & parentPid)
    For Each child In children
        KillTree wmiObj, child.ProcessId
    Next
    Set target = wmiObj.ExecQuery( _
        "SELECT ProcessId FROM Win32_Process WHERE ProcessId = " & parentPid)
    For Each p In target
        On Error Resume Next
        p.Terminate
        On Error GoTo 0
    Next
End Sub
