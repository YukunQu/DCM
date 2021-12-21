function [arttr,metric] = detect_artifacts(dat,S)

%% handle input arguemtns

if nargin<1
    error('Not enough input arguments');
end

if nargin<2
    S = [];
end

if iscell(dat)
    for i = 1:length(dat)
        [arttr{i},metric{i}] = detect_artifacts(dat{i},S);
    end
    return;
end

try method = S.method; catch, method = 'peak2peak'; end
try threshold = S.threshold; catch, threshold = 3; end


%%

nTr = size(dat,1);

%% detect artifacts

switch method
    case 'peak2peak'
        p2p = max(dat,[],2)-min(dat,[],2);
        mnp2p = nanmean(p2p);
        stdp2p = nanstd(p2p);
        arttr = (p2p<(mnp2p-threshold*stdp2p))|(p2p>(mnp2p+threshold*stdp2p));
        
        metric.meas = p2p;
        metric.mnmeas = mnp2p;
        metric.stdmeas = stdp2p;
        
        if 0 %plot artefacts?
            nantr = isnan(p2p);
            subplot(211)
            plot(dat(arttr,:)'); ho; plotmse(dat(~(or(arttr,nantr)),:));
            subplot(212);
            hist(p2p,25); a = axis;
            line([(mnp2p-threshold*stdp2p) (mnp2p-threshold*stdp2p)],[a(3) a(4)],'Color','k','LineWidth',3);
            line([(mnp2p+threshold*stdp2p) (mnp2p+threshold*stdp2p)],[a(3) a(4)],'Color','k','LineWidth',3);
            pause;clf
        end
        
    case 'kurtosis'
        kurt = kurtosis(dat,[],2);
        mnkurt = nanmean(kurt);
        stdkurt = nanstd(kurt);
        arttr = (kurt<(mnkurt-threshold*stdkurt))|(kurt>(mnkurt+threshold*stdkurt));
        
        metric.meas = kurt;
        metric.mnmeas = mnkurt;
        metric.stdmeas = stdkurt;
        
        if 0 %plot artefacts?
            nantr = isnan(kurt);
            subplot(211)
            plot(dat(arttr,:)'); ho; plotmse(dat(~(or(arttr,nantr)),:));
            subplot(212);
            hist(kurt,25); a = axis;
            line([(mnkurt-threshold*stdkurt) (mnkurt-threshold*stdkurt)],[a(3) a(4)],'Color','k','LineWidth',3);
            line([(mnkurt+threshold*stdkurt) (mnkurt+threshold*stdkurt)],[a(3) a(4)],'Color','k','LineWidth',3);
            pause;clf
        end
        
    otherwise
        error('Unrecognised artifact detection method');
end

