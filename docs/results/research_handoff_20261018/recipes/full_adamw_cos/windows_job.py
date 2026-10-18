"""Owned Windows process group: launcher death terminates this worker and its loaders."""
import ctypes
from ctypes import wintypes
class WindowsJob:
 def __init__(self):
  class Basic(ctypes.Structure):
   _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wintypes.DWORD),('min_ws',ctypes.c_size_t),('max_ws',ctypes.c_size_t),('active_limit',wintypes.DWORD),('affinity',ctypes.c_size_t),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
  class IO(ctypes.Structure):_fields_=[(n,ctypes.c_uint64) for n in ['read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes']]
  class Extended(ctypes.Structure):_fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
  self.k=ctypes.WinDLL('kernel32',use_last_error=True)
  self.k.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR];self.k.CreateJobObjectW.restype=wintypes.HANDLE
  self.k.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD];self.k.SetInformationJobObject.restype=wintypes.BOOL
  self.k.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE];self.k.AssignProcessToJobObject.restype=wintypes.BOOL
  self.k.CloseHandle.argtypes=[wintypes.HANDLE]
  self.handle=self.k.CreateJobObjectW(None,None)
  if not self.handle:raise OSError(ctypes.get_last_error(),'CreateJobObject')
  info=Extended();info.basic.flags=0x2000
  if not self.k.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
   self.close();raise OSError(ctypes.get_last_error(),'SetInformationJobObject')
 def assign(self,p):
  if not self.k.AssignProcessToJobObject(self.handle,wintypes.HANDLE(int(p._handle))):raise OSError(ctypes.get_last_error(),'AssignProcessToJobObject')
 def close(self):
  if self.handle:self.k.CloseHandle(self.handle);self.handle=None
