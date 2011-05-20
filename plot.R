
source('~/projects/my_lib/my.fig.R')

df = read.table('heat_distribution', header=T, as.is=T)

my.fig('heat', 1, 1)
plot(0, xlim = range(df$t), ylim = range(df$v1, 0), xlab = 'time', ylab = 'heat', type='n')
lines(df$t, df$v1, lwd=2, lty=1, type='l', col='red')
lines(df$t, df$v2, lwd=2, lty=2, type='l', col='blue')
lines(df$t, df$v3, lwd=2, lty=3, type='l', col='black')
legend('topright', c('node 1', 'node 2', 'node 3,4,5'), lwd = 2, lty = 1:3, col = c('red', 'blue', 'black'), ncol = 3)
my.fig.off()
