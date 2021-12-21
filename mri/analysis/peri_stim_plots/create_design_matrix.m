function [dm,con,conlist] = create_design_matrix(EVdir,subjID,dm_type,badtr,nTr);

if length(badtr)~=nTr
    error('length(badtr) should equal nTr');
end


%% load in logfile
[hd,sd] = get_homedir;

wd = [hd '/projects/stim_act_comp/logfiles_fMRI'];
LFname = sprintf('%s/%s_main.mat',wd,subjID);

LF = load(LFname);
addpath([hd '/projects/stim_act_comp/behaviour_analysis/']);
LF = classify_trials(LF);

%% create design matrix

switch dm_type
    case 'dimensions'
        nEVs = 9;
        dm = nan(nTr,nEVs);
        
        dm(:,1) = ones(nTr,1);
        dm(:,2) = LF.chdimchopt;
        dm(:,3) = LF.chdimunchoptb;
        dm(:,4) = LF.chdimunchoptw;
        dm(:,5) = LF.unchdimchopt;
        dm(:,6) = LF.unchdimunchoptb;
        dm(:,7) = LF.unchdimunchoptw;
        dm(:,8) = LF.went_with==3;
        dm(:,9) = LF.went_with==4;
        to_demean = [];
        to_normalise = [];
        
        con =  eye(size(dm,2)); con(1,:) = []; %everything except main effect
        con(end+1:end+4,:) = [0 1 0 0 -1 0 0 0 0; ...
                             0 0 1 0 0 -1 0 0 0; ...
                             0 1 -1 0 0 0 0 0 0; ...
                             0 0 0 0 1 -1 0 0 0];
        conlist ={'chdimchopt' 'chdimunchoptb' 'chdimunchoptw' ...
                 'unchdimchopt' 'unchdimunchoptb' 'unchdimunchoptw' 'nobrainer' 'went_with_neither' 'cdco-udco' 'cduob-uduob' ...
                 'cdco-cduob' 'udco-uduob'};
        
    case 'dimensions_split'
        nEVs = 11;
        dm = nan(nTr,nEVs);

        dm(:,1) = ones(nTr,1);
        dm(:,2) = LF.chdimchopt.*(LF.went_with==1);
        dm(:,3) = LF.chdimchopt.*(LF.went_with==2);
        dm(:,4) = LF.chdimunchoptb.*(LF.went_with==1);
        dm(:,5) = LF.chdimunchoptb.*(LF.went_with==2);
        dm(:,6) = LF.unchdimchopt.*(LF.went_with==2);
        dm(:,7) = LF.unchdimchopt.*(LF.went_with==1);
        dm(:,8) = LF.unchdimunchoptb.*(LF.went_with==2);
        dm(:,9) = LF.unchdimunchoptb.*(LF.went_with==1);
        dm(:,10) = LF.went_with==3;
        dm(:,11) = LF.went_with==4;
        
        to_demean = [];
        to_normalise =[];
        
        con = eye(size(dm,2)); con(1,:) = []; %everything except main effect
        
        conlist = {'cschdim' 'crchdim' 'usbchdim' 'urbchdim' 'csunchdim' 'crunchdim' ...
            'usbunchdim' 'urbunchdim'  'nobrainer' 'went_with_neither'};

    case 'stim_resp_vals' 
        nEVs = 7;
        dm = nan(nTr,nEVs);
        
        dm(:,1) = ones(nTr,1); %column of ones
        dm(:,2) = LF.chstim;
        dm(:,3) = LF.unchstimb;
        dm(:,4) = LF.unchstimw;
        dm(:,5) = LF.chresp;
        dm(:,6) = LF.unchrespb;
        dm(:,7) = LF.unchrespw;
        
        to_demean = 2:7;
        to_normalise = [];
        
        con(1,:) = [0 1 -1 0 -1 1 0];   conlist{1} = '(chstim-unchstimb)-(chresp-unchrespb)';
        con(2,:) = [0 1 1 0 -1 -1 0];   conlist{2} = '(chstim+unchstimb)-(chresp+unchrespb)';
        con(3,:) = [1 0 0 0 0 0 0];     conlist{3} = 'main effect';
        con(4,:) = [0 1 0 0 0 0 0];     conlist{4} = 'chstim';
        con(5,:) = [0 0 1 0 0 0 0];     conlist{5} = 'unchstimb';
        con(6,:) = [0 0 0 1 0 0 0];     conlist{6} = 'unchstimw';
        con(7,:) = [0 0 0 0 1 0 0];     conlist{7} = 'chresp';
        con(8,:) = [0 0 0 0 0 1 0];     conlist{8} = 'unchrespb';
        con(9,:) = [0 0 0 0 0 0 1];     conlist{9} = 'unchrespw';
        con(10,:) =[0 1 0 0 -1 0 0];   conlist{10} = 'chstim-chresp';

    case 'value_nobrainers2'
        nEVs = 8;
        dm = nan(nTr,nEVs);
        
        dm(:,1) = LF.went_with~=3;
        dm(:,2) = LF.chval.*(LF.went_with~=3);
        dm(:,3) = LF.unchvalb.*(LF.went_with~=3);
        dm(:,4) = LF.unchvalw.*(LF.went_with~=3);
        dm(:,5) = LF.went_with==3;
        dm(:,6) = LF.chval.*(LF.went_with==3);
        dm(:,7) = LF.unchvalb.*(LF.went_with==3);
        dm(:,8) = LF.unchvalw.*(LF.went_with==3);
        
        con = eye(size(dm,2));
        con(end+1,:) = [0 1 0 0 0 -1 0 0];
        con(end+1,:) = [0 0 1 0 0 0 -1 0];
        con(end+1,:) = [1 0 0 0 -1 0 0 0];
        conlist = {'main_eff' 'chval' 'unchvalb' 'unchvalw' 'NB' ...
            'chvalNB' 'unchvalbNB' 'unchvalwNB' 'chvalHard>NB' 'unchvalbHard>NB' 'main_eff>NB'};    
        
        
        to_demean = [];
        to_demean_nonzeros = [];
        to_normalise = [];
        
    case 'simp'
        nEVs = 2;
        dm = nan(nTr,nEVs);
        
        dm(:,1) = LF.went_with~=3;
        dm(:,2) = LF.went_with==3;
        
        con = eye(size(dm,2));
        conlist = {'Hard' 'NB'};
               
        to_demean = [];
        to_normalise = [];
    case 'reward'
        nEVs = 2;
        dm = nan(nTr,nEVs);
        
        dm(:,1) = ones(nTr,1);
        dm(:,2) = load_regressor(sprintf([EVdir '/feedback_reward.txt'],subjID));
        chs = load_regressor(sprintf([EVdir '/decide_chstim.txt'],subjID));
        chr = load_regressor(sprintf([EVdir '/decide_chresp.txt'],subjID));
        dm(:,3) = chs.*chr;
        
        to_demean = [2 3];
        to_normalise = [];
        
        con(1,:) = [0 1 0]; conlist{1} = 'reward-no reward';
        con(2,:) = [0 0 1]; conlist{2} = 'chosen value';
        
    case 'left_right_val'
        nEVs = 4;
        dm = nan(nTr,nEVs);
        
        choseloc = LF.loctr(sub2ind(size(LF.loctr),1:length(LF.loctr),LF.choseopt));
        choseright = LF.loc(choseloc)>0; %compare x position of chosen option to centre
        
        dm(:,1) = 1-choseright;
        dm(:,2) = choseright;
        dm(:,3) = (1-choseright).*LF.chval;
        dm(:,4) = choseright.*LF.chval;
        
        con = eye(4); 
        conlist = {'leftpress' 'rightpress' 'chval_leftpress' 'chval_rightpress'};
        
        to_demean = [];
        to_normalise = [];
        to_demean_nonzeros = [3 4];

    case 'left_right_val_split'
        nEVs = 8;
        dm = nan(nTr,nEVs);
        
        choseloc = LF.loctr(sub2ind(size(LF.loctr),1:length(LF.loctr),LF.choseopt));
        choseright = LF.loc(choseloc)>0; %compare x position of chosen option to centre
        
        dm(:,1) = (1-choseright).*(LF.went_with==1);
        dm(:,2) = choseright.*(LF.went_with==1);
        dm(:,3) = (1-choseright).*(LF.went_with==2);
        dm(:,4) = choseright.*(LF.went_with==2);
        dm(:,5) = (1-choseright).*(LF.went_with==1).*LF.chstim;
        dm(:,6) = choseright.*(LF.went_with==1).*LF.chstim;
        dm(:,7) = (1-choseright).*(LF.went_with==2).*LF.chresp;
        dm(:,8) = choseright.*(LF.went_with==2).*LF.chresp;

        %dm(:,3) = (1-choseright).*LF.chval;
        %dm(:,4) = choseright.*LF.chval;
        
        con = eye(8);
        con(end+1:end+2,:) = [0 0 0 0 -1 0 1 0; ...
                              0 0 0 0 0 -1 0 1];

        conlist = {'leftpress_wws' 'rightpress_wws' 'leftpress_wwr' 'rightpress_wwr' ...
            'leftpress_wws_chstimp' 'rightpress_wws_chstimp' 'leftpress_wwr_chrespp' 'rightpress_wwr_chrespp' ...
            'leftpress_wwrchp>wwschp' 'rightpress_wwrchp>wwschp'};
        
        to_demean = [];
        to_normalise = [];
        
    otherwise
        error('Unrecognised dm_type');
end

%% remove bad trials
dm(badtr,:) = nan;

%% demean/normalise appropriate regressors
dm(:,to_demean) = dm(:,to_demean) - repmat(nanmean(dm(:,to_demean),1),nTr,1);
dm(:,to_normalise) = (dm(:,to_normalise) - repmat(nanmean(dm(:,to_normalise),1),nTr,1))...
    ./repmat(nanstd(dm(:,to_normalise),[],1),nTr,1);
if exist('to_demean_nonzeros','var')
   for i = 1:length(to_demean_nonzeros)
       r = to_demean_nonzeros(i);
       if any(to_demean==r)
           error('cannot have same regressor in to_demean and to_demean_nonzeros');
       end
       nZ = (dm(:,r)~=0)&(~isnan(dm(:,r)));
       dm(nZ,r) = dm(nZ,r) - repmat(mean(dm(nZ,r)),sum(nZ),1);
   end
end

end


%%
function regr = load_regressor(fname)
    tmp = load(fname);
    regr = tmp(:,3);
end

