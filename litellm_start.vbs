Set fso = CreateObject("Scripting.FileSystemObject")
Set wmi = GetObject("winmgmts:\\.\root\cimv2")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pidFile = fso.BuildPath(scriptDir, ".litellm.pid")

' Kill previous instance by PID file (only our own process)
If fso.FileExists(pidFile) Then
    Set f = fso.OpenTextFile(pidFile, 1)
    oldPid = Trim(f.ReadLine)
    f.Close
    KillTree wmi, CLng(oldPid)
    fso.DeleteFile pidFile
    WScript.Sleep 500
End If

' Start hidden via WMI (returns PID)
Set startup = wmi.Get("Win32_ProcessStartup").SpawnInstance_
startup.ShowWindow = 0
Set procClass = wmi.Get("Win32_Process")
result = procClass.Create( _
    "cmd /c """ & fso.BuildPath(scriptDir, "litellm_start.cmd") & """", _
    scriptDir, startup, newPid)

If result = 0 Then
    Set f = fso.CreateTextFile(pidFile, True)
    f.WriteLine newPid
    f.Close
End If

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
