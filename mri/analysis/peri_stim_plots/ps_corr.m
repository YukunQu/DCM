function [dat,badtr] = ps_corr(S);

% ps_corr(S)
%
% fields of S:
% S.subjlist = {'subj102' 'subj103' etc.}; REQUIRED
% S.dm_type = 'stim_resp_vals'/'dimensions2'/'reward'; REQUIRED
% S.maskname; REQUIRED
% S.plotcon; which contrasts to plot, REQUIRED
% S.locking = 'relock' [default]/'decide'/'response'/'feedback'
% S.TR; (in seconds; default 3.01s)
% S.upsrate; (fractions of a TR to upsample to; default 10)
% S.tsdir; directory in which timeseries are stored, with %s as subjID then %s as maskname (default '../%s/masks/%s.txt')
% S.EVdir; directory in which EVs are stored, with %s as subjID (default '../../EVs/%s')
% S.remove_artifacts; default 1 (peak to peak, 3 s.d. away from mean)
% S.group_dm/group_con/gc_names - specifies group level design matrix
% S.bc; to baseline correct data

%% handle input arguments

try subjlist = S.subjlist; catch, error('Must specify S.subjlist'); end
try dm_type = S.dm_type; catch, error('Must specify S.dm_type'); end
try maskname = S.maskname; catch, error('Must specify S.maskname'); end
try plotcon = S.plotcon; catch, error('Must specify S.plotcon'); end
try locking = S.locking; catch, locking = 'relock'; end
try TR = S.TR; catch, TR = 3.01; end
try upsrate = S.upsrate; catch, upsrate = 10; end
try tsdir = S.tsdir; catch, tsdir = '../%s/masks/%s.txt'; end
try EVdir = S.EVdir; catch, EVdir = '../../EVs/%s'; end
try bc = S.bc; catch, bc = 0; end
try nuisance = S.nuisance; catch, nuisance = 0; end
try remove_artifacts = S.remove_artifacts; catch, remove_artifacts = 1; end
nS = length(subjlist); %number of subjects

try
    group_dm = S.group_dm;
catch
    group_dm = ones(nS,1);
    S.group_con = 1;
    S.gc_names = {'Group mean'};
end
try group_con = S.group_con; catch, group_con = eye(size(group_dm,2)); end
try gc_names = S.gc_names; catch, gc_names = cell(size(group_dm,2),1); end

%% load in timeseries

for i = 1:nS
   fname = sprintf(tsdir,subjlist{i},maskname);
   ts{i} = load(fname);
   nV(i) = length(ts{i}); %number of volumes
end

%% load in times

for i = 1:nS
    %load in decide times
    fname = sprintf([EVdir '/decide_onset.txt'],subjlist{i});
    tmp = load(fname); 
    dectimes{i} = tmp(:,1); %times (in s) of decide onset
    decdur{i} = tmp(:,2); %duration (in s) of decision
    meddecdur(i) = median(decdur{i}); %median duration of decision for subject i
    nTr(i,1) = size(tmp,1); %number of trials, based on decision EV
    
    %load in response times (time at which button was pressed, beginning
    %of 4-8s monitor period)
    fname = sprintf([EVdir '/response_onset.txt'],subjlist{i});
    tmp = load(fname); 
    resptimes{i} = tmp(:,1);
    respdur{i} = tmp(:,2);
    medrespdur(i) = median(respdur{i});
    nTr(i,2) = size(tmp,1);
    
    %load in feedback times
    fname = sprintf([EVdir '/feedback_onset.txt'],subjlist{i});
    tmp = load(fname); 
    fbtimes{i} = tmp(:,1);
    fbdur{i} = tmp(:,2);
    medfbdur(i) = median(fbdur{i});
    nTr(i,3) = size(tmp,1);
    
    %load in ITI times
    fname = sprintf([EVdir '/ITI_onset.txt'],subjlist{i});
    tmp = load(fname); 
    ITItimes{i} = tmp(:,1);
    ITIdur{i} = tmp(:,2);
    medITIdur(i) = median(ITIdur{i});
    nTr(i,4) = size(tmp,1);

    if length(unique(nTr(i,:)))~=1
        error('Different numbers of trials in decide/response/feedback/ITI EVs, subject %s',subjlist{i});
    end
end
    
nTr = nTr(:,1); %number of good trials for each subject


%% create upsampled time-locked data matrix    

%find length of time for which to plot data
TR_ups = TR/upsrate; %TR of upsampled data

switch locking
    case 'relock'
        relock_gap = 0; %gap (in seconds) to place between decide-locked timeseries and feedback-locked timeseries
        if relock_gap ~=0
            warning('trying to place gap between decide/feedback epochs - not yet tested');
        end
        dec_baseline = 2.5; %baseline period (in seconds) to place before decide-locked timeseries
        
        dur_declock = ceil((mean(meddecdur)+mean(medrespdur))/TR_ups); %how many (upsampled) samples to include after dec onset
        dur_fblock = dur_declock;  %how many (upsampled) samples to include after feedback onset
        
        event(1).name = 'Decide'; event(1).time = dec_baseline;
        event(2).name = 'Response'; event(2).time = event(1).time+mean(meddecdur);
        event(3).name = 'Feedback'; event(3).time = event(2).time+mean(medrespdur)+relock_gap;
        
        event(4).name = 'ITI'; event(4).time = event(3).time+mean(medfbdur);
        event(5).name = 'Next trial start'; event(5).time = event(4).time+mean(medITIdur);
        
        relock_gap = round(relock_gap/TR_ups); %convert to TRs
        dec_baseline = round(dec_baseline/TR_ups); %convert to TRs
        
    otherwise
        error('not yet implemented')
end


for i = 1:nS
    
    %demean and upsample timeseries
    ts{i}=ts{i}-mean(ts{i}); %demeaned data for subject i
    x=1:nV(i);
    xx=1:(1/upsrate):nV(i);
    ts_ups{i}=spline(x,ts{i},xx); %upsampled demeaned data for subject i
    nV_ups(i) = length(ts_ups{i}); %number of volumes in upsampled data for subject i
    clear x xx;
    
    tt_ups = 0:TR_ups:(nV_ups(i)-1)*TR_ups; %time of each volume, in seconds
    %create data matrix
    switch locking
        case 'relock'
            [decsamp_ups] = findc(tt_ups,dectimes{i}); %decsamp_ups is when dectimes occured in tt_ups
            [fbsamp_ups] = findc(tt_ups,fbtimes{i}); %decsamp_ups is when dectimes occured in tt_ups
            badtr{i} = boolean(zeros(nTr(i),1));
            tind = [];
            for tr = 1:nTr(i)
                %calculate indices of interest in upsampled data: 
                tind(tr,:) = [decsamp_ups(tr)-dec_baseline:decsamp_ups(tr)+dur_declock fbsamp_ups(tr):fbsamp_ups(tr)+dur_fblock];
                if any(tind(tr,:)<1)|any(tind(tr,:)>nV_ups(i))
                    badtr{i}(tr) = true; %indexes trials that stretch outside the sampled timeseries
                end
            end
            
            %create data matrix, leaving nans in place on 'bad' trials (removed later, in regression)
            dat{i} = nan(size(tind));
            dat{i}(~badtr{i},:) = ts_ups{i}(tind(~badtr{i},:));
            
            %baseline correct, if requested
            if bc
                bl = nanmean(dat{i}(:,1:dec_baseline),2); %baseline for each trial
                bl = repmat(bl,1,size(dat{i},2));
                dat{i} = dat{i} - bl;
            end
            
            tt_dat = 0:TR_ups:(size(dat{i},2)-1)*TR_ups;
    end
    
    if remove_artifacts
        Sa = [];
        Sa.method = 'peak2peak';
        arttr{i} = detect_artifacts(dat{i},Sa);
        badtr{i} = badtr{i}|arttr{i};
        dat{i}(arttr{i},:) = nan;
        clear Sa
    end
end

clear *times *dur *samp_ups i tind tmp tr tt_ups


%% create design matrix for each subject

if nuisance
    Sa = [];
    Sa.method = 'peak2peak';
end
for i = 1:nS
    [dm{i},con,conlist] = create_design_matrix(EVdir,subjlist{i},dm_type,badtr{i},nTr(i));
    if nuisance % add in nuisance regressors, if requested
        [arttr,metric] = detect_artifacts(dat{i},Sa);
        %lasttr_p2p = [metric.mnmeas; metric.meas(1:end-1)];
        lasttr_p2p = metric.meas;
        lasttr_p2p(isnan(lasttr_p2p)) = metric.mnmeas;
        dm{i}(:,end+1) = normalise(lasttr_p2p);
        con(:,end+1) = 0;
        con(end+1,end) = 1;
        conlist{end+1} = 'nuisance';
    end
end

%% run regression on each subject

for i = 1:nS
    [c(i,:,:),v(i,:,:),t(i,:,:)] = ols(dat{i}(~badtr{i},:),dm{i}(~badtr{i},:),con);
end

%% do group analysis

nLC = size(con,1); %number of lower level contrasts
nSamp = size(c,3); %number of samples in timeseries
nHC = size(group_con,1); %number of higher level contrasts

c = reshape(c,nS,nLC*nSamp);
[cg,vg,tg] = ols(c,group_dm,group_con);

cg = reshape(cg,nHC,nLC,nSamp);
vg = reshape(vg,nHC,nLC,nSamp);
tg = reshape(tg,nHC,nLC,nSamp);

%% do plotting

for i = 1:nHC
    f(i) = figure;

    %plot relevant contrasts
    plot(tt_dat,squeeze(tg(i,plotcon,:)),'LineWidth',2);
    
    legend(conlist(plotcon));
    
    %reconfigure axes
    %set(gca,'XLim',[-5 30],'XTick',-3:3:25)
    
    %plot events
    xl = get(gca,'XLim');
    line([xl(1) xl(2)],[0 0],'LineWidth',2,'Color','k')
    yl = get(gca,'Ylim');
    for ev = 1:length(event)
        line([event(ev).time event(ev).time],[yl(1) yl(2)]*0.5,'LineWidth',2,'Color','k');
        t = text(event(ev).time,yl(2)*0.55,event(ev).name);
        set(t,'Rotation',45);
    end
    clear t xl yl
    
    xlabel('Time (s)');
    ylabel('T-statistic');
    
    titstr = sprintf('ROI: %s  DM: %s  Group contrast: %s',maskname,dm_type,gc_names{i});
    title(strrep(titstr,'_','\_'));
    
    annotate_tmap(nS-1,gca);
end

