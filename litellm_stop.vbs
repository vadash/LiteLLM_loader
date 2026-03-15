Set wmi = GetObject("winmgmts:\\.\root\cimv2")
Set procs = wmi.ExecQuery( _
    "SELECT ProcessId FROM Win32_Process WHERE " & _
    "CommandLine LIKE '%litellm --config%' OR " & _
    "CommandLine LIKE '%litellm_start.cmd%' OR " & _
    "CommandLine LIKE '%litellm_start_debug.cmd%'")
For Each p In procs
    p.Terminate
Next
