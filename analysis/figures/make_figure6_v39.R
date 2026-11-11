#!/usr/bin/env Rscript
# Figure 6: where the adaptation gain comes from, and what choosing the calibration
# meals is worth. Theme, palette and sizing are carried over unchanged from
# make_figures_v38.R so the visual language of the paper is preserved.
suppressPackageStartupMessages({library(ggplot2);library(patchwork);library(RColorBrewer)})
set.seed(20261111)
NEW <- Sys.getenv('GUTNUT_FIGURE_DATA', 'figure_data')
OUT <- Sys.getenv('GUTNUT_FIGURE_OUT', 'figure')
dir.create(OUT, showWarnings=FALSE)
readnew <- function(n) read.csv(file.path(NEW,paste0(n,'.csv')),check.names=FALSE,stringsAsFactors=FALSE)

INK <- '#253542'; GRID <- '#E5EBEF'; MUTED <- '#7A8993'; PALE <- '#F3F6F8'
TEAL <- brewer.pal(8,'Dark2')[1]; BLUE <- brewer.pal(9,'Blues')[7]
CORAL <- brewer.pal(11,'RdBu')[3]; GREY <- '#9AA7B0'; PLUM <- '#9285AF'
RMSE <- expression(RMSE~(mmol~l^{-1}~h))
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
save <- function(p,name,h) ggsave(file.path(OUT,paste0(name,'.pdf')),p,width=180,height=h,
  units='mm',device=cairo_pdf,family='Arial',bg='white',limitsize=FALSE)
coh <- c(CGMacros='Free-living cohort (44 participants)',PREDICT_UK='Standardized meals (998 participants)')

## ---- a: the adaptation-rule ladder under one fixed population predictor ----
a <- readnew('V39_6a')
a$clipped <- as.character(a$clipped) %in% c('True','TRUE','true','1')
a$cohort_label <- factor(unname(coh[a$cohort]), levels=unname(coh))
a$label <- factor(a$label, levels=rev(unique(a$label[order(a$rank)])))
a$family <- ifelse(a$rule %in% c('shared','eb_intercept'),'Offset or none',
            ifelse(a$rule %in% c('ols_head'),'No shrinkage',
            ifelse(a$rule %in% c('ridge_head','bayes_last_layer'),'Isotropic shrinkage',
                   'Correlated three-coordinate shrinkage')))
famcol <- c('No shrinkage'=CORAL,'Isotropic shrinkage'=GREY,'Offset or none'=BLUE,
            'Correlated three-coordinate shrinkage'=TEAL)
# the unregularised fit is off scale; it is drawn at the panel edge and labelled
cap <- do.call(rbind, lapply(split(a,a$cohort), function(z) {
  lim <- max(z$rmse[!z$clipped]) * 1.14
  z$plotted <- pmin(z$rmse, lim); z$lim <- lim; z }))
pa <- ggplot(cap, aes(plotted, label, fill=family)) +
  geom_segment(aes(x=0, xend=plotted, y=label, yend=label), colour=GRID, linewidth=1.6) +
  geom_point(shape=21, size=2.5, stroke=.6, colour='white') +
  geom_text(data=cap[!cap$clipped,], aes(label=sprintf('%.3f', rmse)),
            hjust=-0.32, size=2.3, colour=INK, family='Arial') +
  geom_text(data=cap[cap$clipped,], aes(label=sprintf('%.2f  (off scale)', rmse)),
            hjust=1.04, size=2.3, colour=CORAL, family='Arial') +
  facet_wrap(~cohort_label, scales='free_x', nrow=1) +
  scale_fill_manual(values=famcol, name=NULL, guide=guide_legend(nrow=2, byrow=TRUE)) +
  scale_x_continuous(expand=expansion(mult=c(0,.22))) +
  labs(x=RMSE, y=NULL) +
  theme(panel.grid.major.y=element_blank(), legend.position='bottom',
        axis.text.y=element_text(size=7.1))

## ---- b: paired contrasts against the reported update ----
b <- readnew('V39_6b')
b$cohort_label <- factor(unname(coh[b$cohort]), levels=unname(coh))
b$reference_label <- factor(b$reference_label,
  levels=rev(unique(b$reference_label[order(b$rank)])))
pb <- ggplot(b, aes(difference, reference_label)) +
  geom_vline(xintercept=0, linetype='dashed', colour=MUTED, linewidth=.4) +
  geom_segment(aes(x=diff_lo, xend=diff_hi, y=reference_label, yend=reference_label),
               linewidth=.9, colour=INK) +
  geom_point(aes(fill=difference>0), shape=21, size=2.2, stroke=.6, colour='white') +
  geom_text(aes(label=sprintf('%d/%d/%d', better, worse, ties)), y=as.numeric(
            factor(b$reference_label))+.34, size=2.05, colour=MUTED, family='Arial', hjust=.5) +
  facet_wrap(~cohort_label, scales='free_x', nrow=1) +
  scale_fill_manual(values=c('FALSE'=CORAL,'TRUE'=TEAL), guide='none') +
  labs(x=expression(RMSE~difference~against~the~reported~update~(mmol~l^{-1}~h)), y=NULL) +
  theme(panel.grid.major.y=element_blank(), axis.text.y=element_text(size=7.1))

## ---- c,d: calibration-set design ----
d <- readnew('V39_6cd')
d$oracle <- as.character(d$oracle) %in% c('True','TRUE','true','1')
d$cohort_label <- factor(unname(coh[d$cohort]), levels=unname(coh))
d$label <- factor(d$label, levels=rev(unique(d$label[order(d$rank)])))
d$cond_label <- ifelse(d$median_cond_number > 1e6, 'singular', sprintf('%.3g', d$median_cond_number))
d$kind <- ifelse(d$oracle,'Oracle upper bound',
          ifelse(d$selector %in% c('first','random'),'Convenience protocol','Composition-based design'))
kcol <- c('Convenience protocol'=BLUE,'Composition-based design'=TEAL,'Oracle upper bound'=PLUM)
pc <- ggplot(d, aes(channel_increment_pct, label, fill=kind)) +
  geom_segment(aes(x=increment_lo_pct, xend=increment_hi_pct, y=label, yend=label),
               linewidth=.9, colour=INK) +
  geom_point(shape=21, size=2.4, stroke=.6, colour='white') +
  facet_wrap(~cohort_label, scales='free_x', nrow=1) +
  scale_fill_manual(values=kcol, name=NULL, guide=guide_legend(nrow=1)) +
  labs(x='Nutrient-channel increment (%)', y=NULL) +
  theme(panel.grid.major.y=element_blank(), legend.position='bottom',
        axis.text.y=element_text(size=7.1))
pd <- ggplot(d, aes(pct_cond_over_100, label, fill=kind)) +
  geom_segment(aes(x=0, xend=pct_cond_over_100, y=label, yend=label), colour=GRID, linewidth=1.6) +
  geom_point(shape=21, size=2.4, stroke=.6, colour='white') +
  geom_text(aes(label=cond_label), hjust=-0.22, size=2.2, colour=MUTED, family='Arial') +
  facet_wrap(~cohort_label, scales='free_x', nrow=1) +
  scale_fill_manual(values=kcol, guide='none') +
  scale_x_continuous(expand=expansion(mult=c(0,.42))) +
  labs(x='Participants with an ill-conditioned calibration design (%); label gives the median condition number', y=NULL) +
  theme(panel.grid.major.y=element_blank(), axis.text.y=element_text(size=7.1))

fig6 <- (tag(pa,'a')/tag(pb,'b')/tag(pc,'c')/tag(pd,'d')) + plot_layout(heights=c(1.16,1.0,1.04,.96))
save(fig6,'Figure_6',250)
cat('Figure_6 written to',OUT,'\n')
