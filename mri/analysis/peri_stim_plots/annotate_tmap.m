function annotate_tmap(dof,ax,pv,twotailed);

% annotate_tmap(dof,ax,pv,twotailed)
%
% dof = degrees of freedom
% ax = axes (default gca)
% pv = pvalues to add (default 0.05 and 0.01)
% twotailed = 1(default) or 0

if nargin<1
   error('must specify degrees of freedom');
end

if nargin<2
    ax = gca;
end

if nargin<3
    pv = [0.05 0.01];
end

if nargin<4
    twotailed = 1;
end

if twotailed
    pv = pv./2;
end

xl = get(ax,'Xlim');

lstyle = {'--',':','-.'}; %line styles

yv = tinv(pv,dof);

for i = 1:length(pv);
    ls = mod(i,3)+1;
    line(xl,[yv(i) yv(i)],'LineStyle',lstyle{ls},'Color','k','LineWidth',0.5);
    line(xl,[-yv(i) -yv(i)],'LineStyle',lstyle{ls},'Color','k','LineWidth',0.5);
end
