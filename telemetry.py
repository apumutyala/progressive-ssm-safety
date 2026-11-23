"""Small Win32 host/process accounting used by the exploratory watchdog.

This is observation, not a Windows Job memory-allocation cap. Functions have no
side effects beyond querying the current host or the supplied child handle.
"""
import ctypes
import os
_kernel=None

class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(name,ctypes.c_uint64) for name in
        ('total','available','total_page','available_page','total_virtual','available_virtual','available_extended')]

class Performance(ctypes.Structure):
    _fields_=[('cb',ctypes.c_uint32)]+[(name,ctypes.c_size_t) for name in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
         'KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(name,ctypes.c_uint32) for name in
        ('HandleCount','ProcessCount','ThreadCount')]

class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_uint32),('PageFaultCount',ctypes.c_uint32)]+[(name,ctypes.c_size_t) for name in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]

def api():
    global _kernel
    if _kernel is not None:return _kernel
    if os.name!='nt':raise RuntimeError('this bounded launcher requires Windows')
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes=[ctypes.POINTER(Memory)];kernel.GlobalMemoryStatusEx.restype=ctypes.c_int
    kernel.K32GetPerformanceInfo.argtypes=[ctypes.POINTER(Performance),ctypes.c_uint32];kernel.K32GetPerformanceInfo.restype=ctypes.c_int
    kernel.K32GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_uint32];kernel.K32GetProcessMemoryInfo.restype=ctypes.c_int
    _kernel=kernel
    return kernel

def host():
    kernel=api();m=Memory();m.length=ctypes.sizeof(m);p=Performance();p.cb=ctypes.sizeof(p)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(m)):raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.K32GetPerformanceInfo(ctypes.byref(p),ctypes.sizeof(p)):raise ctypes.WinError(ctypes.get_last_error())
    return dict(available_physical_bytes=m.available,total_physical_bytes=m.total,
        global_commit_available_bytes=(p.CommitLimit-p.CommitTotal)*p.PageSize,global_commit_limit_bytes=p.CommitLimit*p.PageSize)

def process_memory(handle):
    kernel=api();p=ProcessMemory();p.cb=ctypes.sizeof(p)
    if not kernel.K32GetProcessMemoryInfo(ctypes.c_void_p(handle),ctypes.byref(p),p.cb):raise ctypes.WinError(ctypes.get_last_error())
    return dict(peak_working_set=p.PeakWorkingSetSize,peak_pagefile=p.PeakPagefileUsage)
