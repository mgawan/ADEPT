
import os
import numpy
import pyadept as pa
from pyadept import options as opts

MAX_REF_LEN    =      1200
MAX_QUERY_LEN  =       600
GPU_ID         =         0

MATCH          =  3
MISMATCH       = -3
GAP_OPEN       = -6
GAP_EXTEND     = -1


ref = ["PIAAYKPRSNEILWDGYGVPHIYGVDAPSAFYGYGWAQARSHGDNILRLYGEARGKGAEYWGPDYEQTTVWLLTNGVPERAQQWYAQQSPDFRANLDAFAAGINAYAQQNPDDISPDVRQVLPVSGADVVAHAHRLMNFLYVASPGRTLGEGXSNSWAVAPGKTANGNALLLQNPHLSWTTDYFTYYEAHLVTPDFEIYGATQIGLPVIRFAFNQRMGITNTVNGMVGATNYRLTLQDGGYLYDGQVRPFERPQASYRLRQADGTTVDKPLEIRSSVHGPVFERADGTAVAVRVAGLDRPGMLEQYFDMITADSFDDYEAALARMQVPTFNIVYADREGTINYSFNGVAPKRAEGDIAFWQGLVPGDSSRYLWTETHPLDDLPRVTNPPGGFVQNSNDPPWTPTWPVTYTPKDFPSYLAPQTPHSLRAQQSVRLMSENDDLTLERFMALQLSHRAVMADRTLPDLIPAALIDPDPEVQAAARLLAAWDREFTSDSRAALLFEEWARLFAGQNFAGQAGFATPWSLDKPVSTPYGVRDPKAAVDQLRTAIANTKRKYGAIDRPFGDASRMILNDVNVPGAAGYGNLGSFRVFTWSDPDENGVRTPVHGETWVAMIEFSTPVRAYGLMSYGNSRQPGTTHYSDQIERVSRADFRELLLRREQVEAAVQERTPFNFK", "PIAAYKPRSNEILWDGYGVPHIYGVDAPSAFYGYGWAQARSHGDNILRLYGEARGKGAEYWGPDYEQTTVWLLTNGVPERAQQWYAQQSPDFRANLDAFAAGINAYAQQNPDDISPDVRQVLPVSGADVVAHAHRLMNFLYVASPGRTLGEGXSNSWAVAPGKTANGNALLLQNPHLSWTTDYFTYYEAHLVTPDFEIYGATQIGLPVIRFAFNQRMGITNTVNGMVGATNYRLTLQDGGYLYDGQVRPFERPQASYRLRQADGTTVDKPLEIRSSVHGPVFERADGTAVAVRVAGLDRPGMLEQYFDMITADSFDDYEAALARMQVPTFNIVYADREGTINYSFNGVAPKRAEGDIAFWQGLVPGDSSRYLWTETHPLDDLPRVTNPPGGFVQNSNDPPWTPTWPVTYTPKDFPSYLAPQTPHSLRAQQSVRLMSENDDLTLERFMALQLSHRAVMADRTLPDLIPAALIDPDPEVQAAARLLAAWDREFTSDSRAALLFEEWARLFAGQNFAGQAGFATPWSLDKPVSTPYGVRDPKAAVDQLRTAIANTKRKYGAIDRPFGDASRMILNDVNVPGAAGYGNLGSFRVFTWSDPDENGVRTPVHGETWVAMIEFSTPVRAYGLMSYGNSRQPGTTHYSDQIERVSRADFRELLLRREQVEAAVQERTPFNFK"]

que = ["GIPADNLQSRAKASFDTRVAAAELALNRGVVPSFANGEELLYRNPDPDNTDPSFIASFTKGLPHDDNGAIIDPDDFLAFVRAINSGDEKEIADLTLGPARDPETGLPIWRSDLANSLELEVRGWENSSAGLTFDLEGPDAQSIAMPPAPVLTSPELVAEIAELYLMALGREIEFSEFDSPKNAEYIQFAIDQLNGLEWFNTPAKLGDPPAEIRRRRGEVTVGNLFRGILPGSEVGPYLSQYIIVGSKQIGSATVGNKTLVSPNAADEFDGEIAYGSITISQRVRIATPGRDFMTDLKVFLDVQDAADFRGFESYEPGARLIRTIRDLATWVHFDALYEAYLNACLILLANGVPFDPNLPFQQEDKLDNQDVFVNFGSAHVLSLVTEVATRALKAVRYQKFNIHRRLRPEATGGLISVNKIAAQKGESIFPEVDLAVEELGDILEKAEISNRKQNIADGDPDPDPSFLLPMAFAEGSPFHPSYGSGHAVVAGACVTILKAFFDSGIEIDQVFEVDKDEDKLVKSSFKGTLTVAGELNKLADNIAIGRNMAGVHYFSDQFESLLLGEQVAIGILEEQSLTYGENFFFNLPKFDGTTIQI", "GIPADNLQSRAKASFDTRVAAAELALNRGVVPSFANGEELLYRNPDPDNTDPSFIASFTKGLPHDDNGAIIDPDDFLAFVRAINSGDEKEIADLTLGPARDPETGLPIWRSDLANSLELEVRGWENSSAGLTFDLEGPDAQSIAMPPAPVLTSPELVAEIAELYLMALGREIEFSEFDSPKNAEYIQFAIDQLNGLEWFNTPAKLGDPPAEIRRRRGEVTVGNLFRGILPGSEVGPYLSQYIIVGSKQIGSATVGNKTLVSPNAADEFDGEIAYGSITISQRVRIATPGRDFMTDLKVFLDVQDAADFRGFESYEPGARLIRTIRDLATWVHFDALYEAYLNACLILLANGVPFDPNLPFQQEDKLDNQDVFVNFGSAHVLSLVTEVATRALKAVRYQKFNIHRRLRPEATGGLISVNKIAAQKGESIFPEVDLAVEELGDILEKAEISNRKQNIADGDPDPDPSFLLPMAFAEGSPFHPSYGSGHAVVAGACVTILKAFFDSGIEIDQVFEVDKDEDKLVKSSFKGTLTVAGELNKLADNIAIGRNMAGVHYFSDQFESLLLGEQVAIGILEEQSLTYGENFFFNLPKFDGTTIQI"]

# instantiate a driver object
drv = pa.driver()

# blosum 62 scoring matrix for AA kernels
score_matrix = [4 ,-1 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,0 ,-3 ,-2 ,0 ,-2 ,-1 ,0 ,-4 , -1 ,5 ,0 ,-2 ,
                                            -3 ,1 ,0 ,-2 ,0 ,-3 ,-2 ,2 ,-1 ,-3 ,-2 ,-1 ,-1 ,-3 ,-2 ,-3 ,-1 ,0 ,-1 ,-4 ,
                                            -2 ,0 ,6 ,1 ,-3 ,0 ,0 ,0 ,1 ,-3 ,-3 ,0 ,-2 ,-3 ,-2 ,1 ,0 ,-4 ,-2 ,-3 ,3 ,0 ,-1 ,-4 ,
                                            -2 ,-2 ,1 ,6 ,-3 ,0 ,2 ,-1 ,-1 ,-3 ,-4 ,-1 ,-3 ,-3 ,-1 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
                                            0 ,-3 ,-3 ,-3 ,9 ,-3 ,-4 ,-3 ,-3 ,-1 ,-1 ,-3 ,-1 ,-2 ,-3 ,-1 ,-1 ,-2 ,-2 ,-1 ,-3 ,-3 ,-2 ,-4 ,
                                            -1 ,1 ,0 ,0 ,-3 ,5 ,2 ,-2 ,0 ,-3 ,-2 ,1 ,0 ,-3 ,-1 ,0 ,-1 ,-2 ,-1 ,-2 ,0 ,3 ,-1 ,-4 ,
                                            -1 ,0 ,0 ,2 ,-4 ,2 ,5 ,-2 ,0 ,-3 ,-3 ,1 ,-2 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
                                            0 ,-2 ,0 ,-1 ,-3 ,-2 ,-2 ,6 ,-2 ,-4 ,-4 ,-2 ,-3 ,-3 ,-2 ,0 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-1 ,-4 ,
                                            -2 ,0 ,1 ,-1 ,-3 ,0 ,0 ,-2 ,8 ,-3 ,-3 ,-1 ,-2 ,-1 ,-2 ,-1 ,-2 ,-2 ,2 ,-3 ,0 ,0 ,-1 ,-4 ,
                                            -1 ,-3 ,-3 ,-3 ,-1 ,-3 ,-3 ,-4 ,-3 ,4 ,2 ,-3 ,1 ,0 ,-3 ,-2 ,-1 ,-3 ,-1 ,3 ,-3 ,-3 ,-1 ,-4 ,
                                            -1 ,-2 ,-3 ,-4 ,-1 ,-2 ,-3 ,-4 ,-3 ,2 ,4 ,-2 ,2 ,0 ,-3 ,-2 ,-1 ,-2 ,-1 ,1 ,-4 ,-3 ,-1 ,-4 ,
                                            -1 ,2 ,0 ,-1 ,-3 ,1 ,1 ,-2 ,-1 ,-3 ,-2 ,5 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,0 ,1 ,-1 ,-4 ,
                                            -1 ,-1 ,-2 ,-3 ,-1 ,0 ,-2 ,-3 ,-2 ,1 ,2 ,-1 ,5 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,1 ,-3 ,-1 ,-1 ,-4 ,
                                            -2 ,-3 ,-3 ,-3 ,-2 ,-3 ,-3 ,-3 ,-1 ,0 ,0 ,-3 ,0 ,6 ,-4 ,-2 ,-2 ,1 ,3 ,-1 ,-3 ,-3 ,-1 ,-4 ,
                                            -1 ,-2 ,-2 ,-1 ,-3 ,-1 ,-1 ,-2 ,-2 ,-3 ,-3 ,-1 ,-2 ,-4 ,7 ,-1 ,-1 ,-4 ,-3 ,-2 ,-2 ,-1 ,-2 ,-4 ,
                                            1 ,-1 ,1 ,0 ,-1 ,0 ,0 ,0 ,-1 ,-2 ,-2 ,0 ,-1 ,-2 ,-1 ,4 ,1 ,-3 ,-2 ,-2 ,0 ,0 ,0 ,-4 ,
                                            0 ,-1 ,0 ,-1 ,-1 ,-1 ,-1 ,-2 ,-2 ,-1 ,-1 ,-1 ,-1 ,-2 ,-1 ,1 ,5 ,-2 ,-2 ,0 ,-1 ,-1 ,0 ,-4 ,
                                            -3 ,-3 ,-4 ,-4 ,-2 ,-2 ,-3 ,-2 ,-2 ,-3 ,-2 ,-3 ,-1 ,1 ,-4 ,-3 ,-2 ,11 ,2 ,-3 ,-4 ,-3 ,-2 ,-4 ,
                                            -2 ,-2 ,-2 ,-3 ,-2 ,-1 ,-2 ,-3 ,2 ,-1 ,-1 ,-2 ,-1 ,3 ,-3 ,-2 ,-2 ,2 ,7 ,-1 ,-3 ,-2 ,-1 ,-4 ,
                                            0 ,-3 ,-3 ,-3 ,-1 ,-2 ,-2 ,-3 ,-3 ,3 ,1 ,-2 ,1 ,-1 ,-2 ,-2 ,0 ,-3 ,-1 ,4 ,-3 ,-2 ,-1 ,-4 ,
                                            -2 ,-1 ,3 ,4 ,-3 ,0 ,1 ,-1 ,0 ,-3 ,-4 ,0 ,-3 ,-3 ,-2 ,0 ,-1 ,-4 ,-3 ,-3 ,4 ,1 ,-1 ,-4 ,
                                            -1 ,0 ,0 ,1 ,-3 ,3 ,4 ,-2 ,0 ,-3 ,-3 ,1 ,-1 ,-3 ,-1 ,0 ,-1 ,-3 ,-2 ,-2 ,1 ,4 ,-1 ,-4 ,
                                            0 ,-1 ,-1 ,-1 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-1 ,-2 ,0 ,0 ,-2 ,-1 ,-1 ,-1 ,-1 ,-1 ,-4 ,
                                            -4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,-4 ,1
]

# gap scores
gaps = pa.gap_scores(GAP_OPEN, GAP_EXTEND)

# get max batch size
batch_size = 1 # pa.get_batch_size(GPU_ID, MAX_QUERY_LEN, MAX_REF_LEN, 100)

total_alignments = len(ref)

# print status
print("STATUS: Launching driver", flush=True)

# initialize the driver
all_results = pa.multiGPU(ref, que, opts.ALG_TYPE.SW, opts.SEQ_TYPE.AA, opts.CIGAR.YES, MAX_REF_LEN, MAX_QUERY_LEN, score_matrix, gaps, batch_size)

ts = []
rb = []
re = []
qb = []
qe = []

for i in all_results.results:
    # separate out arrays
    ts.append(i.top_scores())
    rb.append(i.ref_begin())
    re.append(i.ref_end())
    qb.append(i.query_begin())
    qe.append(i.query_end())

# print results
print(ts, rb, re, qb, qe)