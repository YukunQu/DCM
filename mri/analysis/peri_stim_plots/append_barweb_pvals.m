function t = append_barweb_pvals(h,pv,pvcrit,varargin)

% t = append_barweb_pvals(h,pv,pvcrit,varargin)
%
% appends p-value based significance asterisks to bar graph drawn with barweb
%
% INPUT arguments:
% h is handle structure returned by barweb
% pv is array of p-values, one per bar
% pvcrit is criteria for adding asterisks, in increasing levels of
%   significance (default [0.05 0.005 0.0005 0.00005])
% varargin will format text (optional)
%
% OUTPUT argument:
% t is an array of handles, one per set of asterisks
%
% LH 300912

if nargin<3
    pvcrit = [0.05 0.005 0.0005 0.00005];
end

if length(h.bars)~=length(pv)
    error('must have equal number of p-values to number of bars in plot');
end

%% find y-location to plot asterisks

tmp1 = cell2mat(get(h.bars,'YData'))';
mv = max(tmp1(:)+cell2mat(get(h.errors,'Ldata')));
yl = get(h.ax,'Ylim');
set(h.ax,'Ylim',[yl(1) yl(2)*1.3]);
yl = (yl(2)*1.3 + mv)/2;

%% plot asterisks

for i = 1:length(pv)
    ts = repmat('*',1,sum(pv(i)<pvcrit)); %text string for this p-value
    tmp = get(get(h.bars(i),'Children'),'Xdata');
    xl = (tmp(1) + tmp(3))./2; %x-location for this bar
    t(i) = text(xl,yl,ts,'HorizontalAlignment','center',varargin{:});
end