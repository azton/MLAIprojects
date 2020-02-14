import os


def adjust_paths():
    for file in ['test','val','train']:
        j_id = os.getenv('SLURM_JOB_ID')
        prepath = '/oasis/scratch/comet/azton/temp_project/MLAIprojects'
        f = open('orig_%s.csv'%file, 'r')
        fout = open('%s.csv'%file, 'w')
        fout.write('images,labels\n')
        for l in f:
            if '/' in l:
                l = l.split(',')
                inp = l[0]
                label = l[1]
                inputs = '%s%s'\
                    %(prepath,inp)
                labels = '%s%s'\
                    %(prepath, label)
                fout.write('%s,%s\n'%(inputs, labels))
        fout.close()
        f.close()

