Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = fso.GetParentFolderName(WScript.ScriptFullName)

' Kill any existing litellm processes (only one copy allowed)
Set wmi = GetObject("winmgmts:\\.\root\cimv2")
Set procs = wmi.ExecQuery( _
    "SELECT ProcessId FROM Win32_Process WHERE " & _
    "CommandLine LIKE '%litellm --config%' OR " & _
    "CommandLine LIKE '%litellm_start.cmd%' OR " & _
    "CommandLine LIKE '%litellm_start_debug.cmd%'")
For Each p In procs
    p.Terminate
Next

WScript.Sleep 500

' Start hidden (no window)
WshShell.Run Chr(34) & "litellm_start.cmd" & Chr(34), 0
