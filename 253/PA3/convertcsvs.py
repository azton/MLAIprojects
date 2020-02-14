

prepath = '/oasis/scratch/comet/azton/temp_project/MLAIprojects/data'
f = open('orig_test.csv', 'r')
fout = open('test.csv', 'w')
fout.write('images,labels\n')
for l in f:
    if '/' in l:
        l = l.split(',')
        inp = l[0].split('/')
        label = l[1].split('/')
        inputs = '%s/%s/%s/%s/%s/%s'\
            %(prepath,inp[2], inp[3], inp[4], inp[5], inp[6])
        labels = '%s/%s/%s/%s/%s/%s'\
            %(prepath, label[2], label[3], label[4], label[5], label[6])
        fout.write('%s,%s\n'%(inputs, labels))
fout.close()
f.close()

f = open('orig_train.csv', 'r')
fout = open('train.csv', 'w')
fout.write('images,labels\n')
for l in f:
    if '/' in l:
        l = l.split(',')
        inp = l[0].split('/')
        label = l[1].split('/')
        inputs = '%s/%s/%s/%s/%s/%s'\
            %(prepath,inp[2], inp[3], inp[4], inp[5], inp[6])
        labels = '%s/%s/%s/%s/%s/%s'\
            %(prepath, label[2], label[3], label[4], label[5], label[6])
        fout.write('%s,%s\n'%(inputs, labels))
fout.close()
f.close()

f = open('orig_val.csv', 'r')
fout = open('val.csv', 'w')
fout.write('images,labels\n')
for l in f:
    if '/' in l:
        l = l.split(',')
        inp = l[0].split('/')
        label = l[1].split('/')
        inputs = '%s/%s/%s/%s/%s/%s'\
            %(prepath, inp[2], inp[3], inp[4], inp[5], inp[6])
        labels = '%s/%s/%s/%s/%s/%s'\
            %(prepath,label[2], label[3], label[4], label[5], label[6])
        fout.write('%s,%s\n'%(inputs, labels))
fout.close()
f.close()
