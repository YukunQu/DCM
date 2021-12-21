function fields_to_workspace(structure)

% fields_to_workspace(struct)
%
% adds all fields from the structure struct to the workspace
% will overwrite variables that already have corresponding field!
%
% LH 081109

if ~isstruct(structure)
    error('Input argument to fields_to_workspace must be structure');
end

names = fieldnames(structure);
for i = 1:length(names)
    assignin('caller',names{i},getfield(structure,names{i}));
end