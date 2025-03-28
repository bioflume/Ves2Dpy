function [options,prams] = initVes2D(options,prams)
% Set a path pointing to src directory and set options and
% prams to default values if they are left blank


PramList = {'N','nv','T','dt','Nbd','nvbd','kappa',...
    'viscCont','gmresTol','gmresMaxIter','tstepTol','areaLenTol',...
    'minDist','repStrength','farFieldSpeed'};
defaultPram.N = 64;
defaultPram.nv = 1;
defaultPram.Nbd = 0;
defaultPram.nvbd = 0;
defaultPram.T = 1;
defaultPram.dt = 1E-5;
defaultPram.kappa = 1e-1;
defaultPram.viscCont = 1;
defaultPram.gmresTol = 1e-12;
defaultPram.gmresMaxIter = 200;
defaultPram.tstepTol = 1e-2;
defaultPram.areaLenTol = 1e-2;
defaultPram.repStrength = 900;
defaultPram.minDist = 0.4;
defaultPram.farFieldSpeed = 1;

for k = 1:length(PramList)
  if ~isfield(prams,PramList{k})
    eval(['prams.' PramList{k} '=defaultPram.' PramList{k} ';'])
    % Set any unassigned parameters to a default value
  end
end

OptList = {'farField','repulsion','correctShape','reparameterization',...
    'filterShape','usePreco','matFreeWalls','confined'};
defaultOpt.farField = 'shear';
defaultOpt.repulsion = false;
defaultOpt.correctShape = false;
defaultOpt.reparameterization = false;
defaultOpt.usePreco = true;
defaultOpt.matFreeWalls = false;
defaultOpt.confined = false;

for k = 1:length(OptList)
  if ~isfield(options,OptList{k})
    eval(['options.' OptList{k} '=defaultOpt.' OptList{k} ';'])
    % Set any unassigned options to a default value
  end
end

% If the geometry is unbounded, make sure to set the number
% of points and number of components of the solid walls
% to 0.  Otherwise, later components will crash
if ~options.confined
  prams.Nbd = 0;
  prams.nvbd = 0;
end

if numel(prams.viscCont) ~=prams.nv
  prams.viscCont = prams.viscCont*ones(1,prams.nv);
end





