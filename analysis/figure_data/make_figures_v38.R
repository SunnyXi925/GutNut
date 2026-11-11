#!/usr/bin/env Rscript
# Revised Figure 2 and Figure 3 for v38. Theme, palette and helpers are carried over
# unchanged from make_figures_v17.R so the visual language of the paper is preserved.
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(ggbeeswarm);library(RColorBrewer)})
set.seed(20261018)
OLD <- '/Users/fengxi.25/Downloads/1017GutNut/source_data'
NEW <- '/Users/fengxi.25/Desktop/GMNPS/GutNut_NCS_evidence_2026-11-08/figure_data'
OUT <- '/Users/fengxi.25/Desktop/GMNPS/GutNut_NCS_evidence_2026-11-08/figure'
dir.create(OUT, showWarnings=FALSE)
readold <- function(n) read.csv(file.path(OLD,paste0(n,'.csv')),check.names=FALSE,stringsAsFactors=FALSE)
readnew <- function(n) read.csv(file.path(NEW,paste0(n,'.csv')),check.names=FALSE,stringsAsFactors=FALSE)

INK <- '#253542'; GRID <- '#E5EBEF'; MUTED <- '#7A8993'; PALE <- '#F3F6F8'
TEAL <- brewer.pal(8,'Dark2')[1]; BLUE <- brewer.pal(9,'Blues')[7]
CORAL <- brewer.pal(11,'RdBu')[3]; LIPID <- brewer.pal(8,'Set2')[2]
DIV <- c(brewer.pal(11,'RdBu')[3:5], '#FAFBFB', brewer.pal(11,'BrBG')[7:9])
SEQ <- colorRampPalette(brewer.pal(9,'Blues')[2:9])(128)
RMSE <- expression(RMSE~(mmol~l^{-1}~h))
PROFILE <- expression(Profile~RMSE~(mmol~l^{-1}~h))
GAIN <- expression(RMSE~reduction~(mmol~l^{-1}~h))
theme_set(theme_minimal(base_size=8.2,base_family='Arial') + theme(
  text=element_text(colour=INK),axis.text=element_text(size=7.2,colour=INK),axis.title=element_text(size=7.8),
  panel.grid.minor=element_blank(),panel.grid.major=element_line(colour=GRID,linewidth=.28),
  axis.line=element_blank(),axis.ticks=element_blank(),
  plot.title=element_blank(),plot.subtitle=element_blank(),
  plot.tag=element_text(size=10.8,face='bold',colour='#172B39'),plot.tag.position='topleft',
  plot.margin=margin(5,6,5,5),legend.title=element_text(size=7),legend.text=element_text(size=6.8),
  legend.key.height=grid::unit(2.8,'mm'),legend.key.width=grid::unit(5,'mm'),legend.margin=margin(0,0,0,0),
  legend.box.spacing=grid::unit(1,'mm'),strip.background=element_rect(fill=PALE,colour=NA),
  strip.text=element_text(size=7.5,colour=INK),panel.spacing=grid::unit(2.5,'mm')))
tag <- function(p,t) p+labs(tag=t)
heat_theme <- theme(panel.grid=element_blank(),axis.line=element_blank())

half_density <- function(d,amplitude=.35) {
  do.call(rbind,lapply(split(d,d$group),function(z) {
    xx <- z$value; row <- unique(z$row); col <- unique(z$colour)
    if(length(unique(xx))<2) return(data.frame(x=xx[1],y=row,group=z$group[1],colour=col))
    dd <- density(xx,from=min(xx),to=max(xx),n=160,adjust=1)
    data.frame(x=c(dd$x[1],dd$x,tail(dd$x,1)),y=c(row+.045,row+.045+amplitude*dd$y/max(dd$y),row+.045),
               group=z$group[1],colour=col)
  }))
}
cloud <- function(d,ag,labels,xlab,show_values=TRUE,digits=3,xlimits=NULL,amp=.35,psize=.83) {
  dens <- half_density(d,amp)
  n <- length(labels); xmax <- max(c(d$value,ag$hi),na.rm=TRUE); xmin <- min(c(d$value,ag$lo),na.rm=TRUE)
  span <- xmax-xmin; if(is.null(xlimits)) xlimits <- c(xmin-span*.035,xmax+span*if(show_values).18 else .04)
  stripes <- data.frame(row=seq_len(n))
  p <- ggplot()+geom_rect(data=stripes[stripes$row%%2==1,,drop=FALSE],
      aes(xmin=-Inf,xmax=Inf,ymin=row-.42,ymax=row+.45),fill=PALE,colour=NA)+
    geom_polygon(data=dens,aes(x,y,group=group,fill=colour),alpha=.40,colour=NA)+
    geom_quasirandom(data=d,aes(value,row-.16,group=group,colour=colour),orientation='y',width=.07,size=psize,alpha=.70)+
    geom_segment(data=ag,aes(x=lo,xend=hi,y=row,yend=row),linewidth=.9,colour=INK)+
    geom_point(data=ag,aes(value,row,fill=colour),shape=21,size=2.1,stroke=.65,colour='white')+
    scale_fill_identity()+scale_colour_identity()+
    scale_y_continuous(breaks=1:n,labels=rev(labels),limits=c(.45,n+.5),expand=c(0,0))+
    scale_x_continuous(limits=xlimits,expand=c(0,0))+labs(x=xlab,y=NULL)+
    theme(panel.grid.major.y=element_blank(),axis.text.y=element_text(size=7.2))
  if(show_values) p<-p+geom_text(data=ag,aes(x=xlimits[2],y=row,label=sprintf(paste0('%.',digits,'f'),value)),
      hjust=1,size=2.45,colour=INK,family='Arial')
  p
}
save <- function(p,name,h) {
  ggsave(file.path(OUT,paste0(name,'.pdf')),p,width=180,height=h,units='mm',device=cairo_pdf,family='Arial',bg='white',limitsize=FALSE)
}

## ---------------- FIGURE 2 ----------------
# a: calibration curve now runs to eleven meals so the plateau is visible (annotation p21)
hist_levels <- c('Shared','0','1','2','3','5','7','9','11')
hist_labels <- c('Shared predictor','Prior only','1 meal','2 meals','3 meals','5 meals','7 meals','9 meals','11 meals')
ramp <- colorRampPalette(c('#BFE0D6',TEAL,'#1F7A63'))(7)
hc <- c(Shared=BLUE,'0'='#C3CED6'); for(i in seq_along(c('1','2','3','5','7','9','11'))) hc[c('1','2','3','5','7','9','11')[i]] <- ramp[i]
h <- readnew('V38_2a'); h$history <- as.character(h$history)
h$group <- h$history; h$row <- length(hist_levels)+1-match(h$history,hist_levels)
h$value <- sqrt(h$mse); h$colour <- unname(hc[h$history])
ag <- readnew('V38_2a_intervals'); ag$model <- as.character(ag$model)
ag$row <- length(hist_levels)+1-match(ag$model,hist_levels); ag$colour <- unname(hc[ag$model])
p2a <- cloud(h,ag,hist_labels,RMSE,amp=.30,psize=.70)

# b: the y-axis title sat too far from the axis (annotation p21)
shared <- readold('V12_2c'); cal <- readold('V12_2d'); cal <- cal[match(shared$meal_event_id,cal$meal_event_id),]
stopifnot(identical(shared$meal_event_id,cal$meal_event_id))
ecdf2 <- function(d,name) {
  err <- abs(d$prediction-d$target); nn <- table(d$participant_id)
  ww <- 1/(length(nn)*as.numeric(nn[d$participant_id])); o <- order(err)
  data.frame(model=name,threshold=c(0,err[o]),probability=c(0,100*cumsum(ww[o])))
}
cdf <- rbind(ecdf2(shared,'Shared'),ecdf2(cal,'GutNut'))
cdf <- rbind(cdf,data.frame(model=c('Shared','GutNut'),threshold=max(cdf$threshold),probability=100))
cdf <- cdf[order(cdf$model,cdf$threshold,cdf$probability),]
gridx <- sort(unique(cdf$threshold))
interp <- function(name) approx(cdf$threshold[cdf$model==name],cdf$probability[cdf$model==name],xout=gridx,method='constant',f=0,ties=max,rule=2)$y
band <- data.frame(x=gridx,shared=interp('Shared'),gutnut=interp('GutNut'))
p2b <- ggplot(cdf,aes(threshold,probability,colour=model,linetype=model))+
  geom_ribbon(data=band,aes(x=x,ymin=pmin(shared,gutnut),ymax=pmax(shared,gutnut)),inherit.aes=FALSE,fill=TEAL,alpha=.14)+
  geom_step(linewidth=.8)+scale_colour_manual(values=c(Shared=BLUE,GutNut=TEAL))+
  scale_linetype_manual(values=c(Shared='22',GutNut='solid'))+
  scale_x_continuous(expand=c(0,0),breaks=c(0,3,6,9))+
  scale_y_continuous(limits=c(0,100),breaks=c(0,25,50,75,100),expand=expansion(mult=c(0,.02)))+
  labs(x=expression(Absolute~error~(mmol~l^{-1}~h)),y='Cumulative share (%)',colour=NULL,linetype=NULL)+
  theme(legend.position='inside',legend.position.inside=c(.69,.28),
        axis.title.y=element_text(margin=margin(r=1.2)),axis.text.y=element_text(margin=margin(r=.8)),
        plot.margin=margin(5,6,5,2))

# c: per-participant gain, now spanning the full width so panel order reads a, b, c
g <- readnew('V38_2c'); g$history <- factor(g$history,levels=c('1','3','11'))
rng <- aggregate(gain~row,g,function(x) c(min=min(x),max=max(x)))
rng$lo <- rng$gain[,1]; rng$hi <- rng$gain[,2]
p2c <- ggplot(g,aes(gain,row))+geom_vline(xintercept=0,linetype='dashed',colour=MUTED,linewidth=.45)+
  geom_segment(data=rng,aes(x=lo,xend=hi,y=row,yend=row),inherit.aes=FALSE,colour='#BEC8CF',linewidth=.45)+
  geom_point(aes(colour=history,shape=history),size=1.55,stroke=.6)+
  scale_colour_manual(values=c('1'=BLUE,'3'=TEAL,'11'=LIPID),labels=c('1 meal','3 meals','11 meals'))+
  scale_shape_manual(values=c('1'=1,'3'=16,'11'=17),labels=c('1 meal','3 meals','11 meals'))+
  scale_y_reverse(breaks=c(1,11,22,33,44),expand=expansion(add=1))+
  labs(x=GAIN,y='Participants',colour=NULL,shape=NULL)+
  theme(legend.position='top',panel.grid.major.y=element_blank())

joint_prediction <- function(d,name,rmse,tag_letter) {
  L <- c(-1.5,14.5); breaks <- seq(-1.5,14.5,.5)
  main <- ggplot(d,aes(target,prediction))+geom_hex(binwidth=c(.50,.50))+
    geom_abline(slope=1,intercept=0,linetype='dashed',colour=MUTED,linewidth=.4)+
    scale_fill_gradientn(colours=SEQ,trans='log10',limits=c(1,100),breaks=c(1,10,100),name='Meals',
                         guide=guide_colourbar(barwidth=grid::unit(24,'mm'),barheight=grid::unit(2.6,'mm')))+
    scale_x_continuous(limits=L,expand=c(0,0),breaks=c(0,5,10))+scale_y_continuous(limits=L,expand=c(0,0),breaks=c(0,5,10))+
    labs(x=expression(Observed~(mmol~l^{-1}~h)),y=expression(Predicted~(mmol~l^{-1}~h)))+
    annotate('text',x=-.6,y=13.8,label=paste(name,paste0('RMSE ',rmse),sep='\n'),hjust=0,vjust=1,size=2.65,family='Arial',colour=INK)+
    theme(legend.position='bottom')
  top <- ggplot(d,aes(target))+geom_histogram(breaks=breaks,fill='#ABC7DC',colour=NA)+
    scale_x_continuous(limits=L,expand=c(0,0))+scale_y_continuous(expand=c(0,0))+
    theme_void()+theme(plot.margin=margin(1,2,0,2),plot.tag=element_text(size=10.8,face='bold',colour=INK),plot.tag.position='topleft')+labs(tag=tag_letter)
  right <- ggplot(d,aes(prediction))+geom_histogram(breaks=breaks,fill=if(name=='GutNut')'#7BBFAC' else '#ABC7DC',colour=NA)+
    scale_x_continuous(limits=L,expand=c(0,0))+scale_y_continuous(expand=c(0,0))+coord_flip()+
    theme_void()+theme(plot.margin=margin(2,1,2,0))
  wrap_plots(A=top,B=main,C=right,design='A#\nBC',widths=c(1,.15),heights=c(.14,1))
}
p2d <- joint_prediction(shared,'Shared','2.364','d'); p2e <- joint_prediction(cal,'GutNut','1.977','e')
fig2 <- ((tag(p2a,'a')|tag(p2b,'b'))/tag(p2c,'c')/(p2d|p2e))+plot_layout(heights=c(1.35,.95,1.05))
save(fig2,'Figure_2',208)

## ---------------- FIGURE 3 ----------------
intervals <- readold('V12_aggregate_RMSE_intervals')
m <- readold('V12_3a'); mods <- c('full','offset_fat','offset_fibre','nutrients_only','offset_only')
names_m <- c('GutNut','No MAC','No LIPID','No intercept','Intercept only')
mc <- c(full=TEAL,offset_fat='#E89B82',offset_fibre='#D5B45B',nutrients_only='#9285AF',offset_only=BLUE)
model_cloud <- function(metric,source_panel) {
  d <- m; d$row <- 6-match(d$model,mods); d$group <- d$model; d$value <- sqrt(d[[metric]]); d$colour <- unname(mc[d$model])
  a <- intervals[intervals$panel==source_panel,]; a$row <- 6-match(a$model,mods); a$colour <- unname(mc[a$model])
  cloud(d,a,names_m,if(metric=='mse') RMSE else PROFILE)
}
p3a <- model_cloud('mse','3a'); p3b <- model_cloud('profile_mse','3b')

# c: GutNut itself is now drawn as the reference, with its own values on the panel (annotation p22)
ref <- readnew('V38_3c_reference')
eff <- readold('module_effects'); eff <- eff[eff$metric %in% c('rmse','profile_rmse'),]
eff$row <- 5-match(eff$reference,mods[-1])
eff$view <- factor(eff$metric,levels=c('rmse','profile_rmse'),labels=c('Absolute','Within-person'))
eff$pos <- eff$row+ifelse(eff$metric=='rmse',.13,-.13)
xr <- range(c(eff$ci_low,eff$ci_high,0)); xr <- c(min(0,xr[1])-.012, xr[2]+.055)
p3c <- ggplot(eff,aes(gain,pos,colour=view,shape=view))+
  annotate('rect',xmin=xr[1],xmax=0,ymin=.5,ymax=4.5,fill=TEAL,alpha=.07)+
  geom_vline(xintercept=0,linetype='solid',colour=TEAL,linewidth=.7)+
  geom_errorbar(aes(xmin=ci_low,xmax=ci_high),orientation='y',width=.08,linewidth=.65)+geom_point(size=2.1)+
  annotate('text',x=0,y=4.46,label=sprintf('GutNut reference  RMSE %.3f, within-person %.3f',ref$rmse[1],ref$profile_rmse[1]),
           hjust=-0.02,vjust=1,size=2.3,family='Arial',colour=TEAL)+
  scale_colour_manual(values=c(Absolute=BLUE,'Within-person'=TEAL))+scale_shape_manual(values=c(Absolute=16,'Within-person'=18))+
  scale_y_continuous(breaks=1:4,labels=rev(names_m[-1]),limits=c(.5,4.5))+
  scale_x_continuous(limits=xr,breaks=c(0,.1,.2,.3),expand=c(0,0))+
  labs(x=expression(Ablation~minus~GutNut~RMSE),y=NULL,colour=NULL,shape=NULL)+
  theme(legend.position='top',panel.grid.major.y=element_blank())

decomp <- readold('V9_S3a')
d <- rbind(data.frame(model=decomp$model,component='Within-person',value=decomp$profile),
           data.frame(model=decomp$model,component='Mean response',value=decomp$level))
d$row <- 4-match(d$model,c('Shared','Intercept','Full')); d$component <- factor(d$component,levels=c('Mean response','Within-person'))
totals <- aggregate(value~row,d,sum)
p3d <- ggplot(d,aes(row,value,fill=component))+geom_col(width=.48,colour='white',linewidth=.5)+
  scale_fill_manual(values=c('Within-person'=BLUE,'Mean response'='#B8CDD9'))+
  geom_text(data=totals,aes(row,value+.08,label=sprintf('%.2f',value)),inherit.aes=FALSE,hjust=0,size=2.45,family='Arial',colour=INK)+
  scale_x_continuous(breaks=1:3,labels=c('GutNut','Intercept only','Shared'),expand=expansion(add=.55))+
  scale_y_continuous(limits=c(0,max(totals$value)*1.16),expand=c(0,0))+coord_flip()+
  labs(x=NULL,y=expression(MSE~((mmol~l^{-1}~h)^2)),fill=NULL)+
  theme(legend.position='top',panel.grid.major.y=element_blank())+guides(fill=guide_legend(nrow=1))

q <- readold('V12_3c'); q$history <- factor(q$history_k,levels=c(0,1,3,5)); lim <- max(abs(q$gain))
p3e <- ggplot(q,aes(row,history,fill=gain))+geom_tile(colour='white',linewidth=.16)+
  scale_fill_gradientn(colours=DIV,limits=c(-lim,lim),name='RMSE reduction',
                       guide=guide_colourbar(barwidth=grid::unit(32,'mm'),barheight=grid::unit(2.8,'mm')))+
  scale_x_continuous(breaks=c(1,11,22,33,44),expand=c(0,0))+scale_y_discrete(limits=rev,expand=c(0,0))+
  labs(x='Participants',y='Calibration meals')+heat_theme+theme(legend.position='bottom')

# f: every prior arm now carries its permutation reference band (annotation p22)
q2 <- readold('V12_3d'); refs <- c('matched_uniform','matched_host_prior','no_micro')
q2$row <- 4-match(q2$reference,refs); q2$group <- q2$reference; q2$value <- q2$individual_gain
cols <- c(matched_uniform=TEAL,matched_host_prior=BLUE,no_micro='#9285AF'); q2$colour <- unname(cols[q2$reference])
a2 <- unique(q2[,c('reference','row','aggregate_gain','ci_low','ci_high','colour')])
a2$value <- a2$aggregate_gain; a2$lo <- a2$ci_low; a2$hi <- a2$ci_high
perm <- readnew('V38_3f_permutation'); perm$row <- 4-match(perm$reference,refs)
p3f <- cloud(q2,a2,c('Population prior','Host prior','Tuned non-microbial'),GAIN,show_values=FALSE)+
  geom_rect(data=perm,aes(xmin=perm_lo,xmax=perm_hi,ymin=row-.40,ymax=row+.44),
            inherit.aes=FALSE,fill='#B0453A',alpha=.13)+
  geom_segment(data=perm,aes(x=perm_median,xend=perm_median,y=row-.40,yend=row+.44),
               inherit.aes=FALSE,colour='#B0453A',linewidth=.5,linetype='21')+
  geom_vline(xintercept=0,linetype='dashed',colour=MUTED,linewidth=.4)+
  annotate('text',x=max(q2$value)*.98,y=3.46,label='Permuted microbiome reference',
           hjust=1,vjust=1,size=2.25,family='Arial',colour='#B0453A')
fig3 <- ((tag(p3a,'a')|tag(p3b,'b'))/(tag(p3c,'c')|tag(p3d,'d'))/(tag(p3e,'e')|tag(p3f,'f')))+
  plot_layout(heights=c(1.08,.9,.82))
save(fig3,'Figure_3',181)
cat('figures written to',OUT,'\n')
