

f = open('orig_test.csv', 'r')
fout = open('test.csv', 'w')
fout.write('images,labels\n')
for l in f:
    if '/' in l:
        l = l.split(',')
        inp = l[0].split('/')
        label = l[1].split('/')
        inputs = 'D:\MLAIprojects\data\%s\%s\%s\%s\%s'\
            %(inp[2], inp[3], inp[4], inp[5], inp[6])
        labels = 'D:\MLAIprojects\data\%s\%s\%s\%s\%s'\
            %(label[2], label[3], label[4], label[5], label[6])
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
        inputs = 'D:\MLAIprojects\data\%s\%s\%s\%s\%s'\
            %(inp[2], inp[3], inp[4], inp[5], inp[6])
        labels = 'D:\MLAIprojects\data\%s\%s\%s\%s\%s'\
            %(label[2], label[3], label[4], label[5], label[6])
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
        inputs = 'D:\MLAIprojects\data\%s\%s\%s\%s\%s'\
            %(inp[2], inp[3], inp[4], inp[5], inp[6])
        labels = 'D:\MLAIprojects\data\%s\%s\%s\%s\%s'\
            %(label[2], label[3], label[4], label[5], label[6])
        fout.write('%s,%s\n'%(inputs, labels))
fout.close()
f.close()
