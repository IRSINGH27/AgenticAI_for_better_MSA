def reader(file:str):
    import re
    result={}
    with open(file,'r') as f:
        for i in f.readlines():
            if i.startswith('>'):
                key=i.strip().split('>')[1]
                pfam=re.findall(r'PF\d+',key)[0]
                if pfam not in result.keys():
                    result[pfam]={}
                result[pfam].update({key:''})
            else:
                x=result[pfam][key]
                x+=i.strip()
                result[pfam][key]=x
    print('file Read')
    return result

def writer(outputFolder:str,pfam:str,seqDict:dict):
    from os import path
    output=path.join(outputFolder,pfam+'.faa')
    with open(output,'w')as f:
        for header,seq in seqDict.items():
            f.write(f'>{header}\n{seq}\n')
    return None

def worker(inputFile:str,outputFolder:str,ncore:int=1):
    from multiprocessing import Pool
    from os import makedirs
    from os import path
    if not path.exists(outputFolder):
        makedirs(outputFolder)
    pfamSeqDict=reader(inputFile)
    arguments=[(outputFolder,pfam,seqDict) for pfam,seqDict in pfamSeqDict.items()]
    with Pool(ncore) as pool:
        print('writing Start')
        pool.starmap(writer,arguments)
    return None

if __name__=='__main__':
    import pathlib
    SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
    PROJECT_ROOT = SCRIPT_DIR.parent
    __PFAMFILE__ = str(PROJECT_ROOT / "data" / "pfam" / "Pfam-A.fasta")
    __OUTPUTFOLDER__ = str(PROJECT_ROOT / "data" / "pfam" / "families")
    worker(__PFAMFILE__,__OUTPUTFOLDER__,50)