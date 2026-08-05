import struct, os, glob, sys
def stored_eof(p):
    with open(p,'rb') as f: h=f.read(64)
    if h[:8]!=b'\x89HDF\r\n\x1a\n': return None
    v=h[8]
    if v in (0,1): return struct.unpack_from('<Q',h,40 if v==0 else 48)[0]
    if v in (2,3): return struct.unpack_from('<Q',h,28)[0]
    return None
for d in sys.argv[1:]:
    ok,bad=[],[]
    for p in sorted(glob.glob(os.path.join(d,'*','data.hdf5'))):
        n=os.path.basename(os.path.dirname(p)); a=os.path.getsize(p); e=stored_eof(p)
        if e and a>=e: ok.append(n)
        else: bad.append('%s (have %d, need %s)'%(n,a,e))
    print('=== %s  OK:%d BAD:%d'%(d,len(ok),len(bad)))
    for b in bad: print('   ',b)
