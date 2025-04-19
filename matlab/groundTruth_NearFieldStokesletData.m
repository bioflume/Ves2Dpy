function groundTruth_NearFieldStokesletData(X)

oc = curve_py;

% Create 4 layers inside/outside vesicle
% layer on the vesicle is separate
nlayers = 4; 

% num. points
N = numel(X(:,1))/2;
nves = numel(X(1,:));
op = poten_py(N);


% build basis 
theta = (0:N-1)'/N*2*pi;
ks = (0:N-1)';
basis = 1/N*exp(1i*theta*ks');
Br = real(basis); Bi = imag(basis);


for ives = 1 : nves
  
  % Build vesicle
  vesicle = capsules_py(X(:,ives),[],[],1,1);
  
  % Generate the grid for velocity
  [~,tang] = oc.diffProp(X(:,ives));
  % get x and y components of normal vector at each point
  nx = tang(N+1:2*N);
  ny = -tang(1:N);

  % Points where velocity is calculated involve the points on vesicle
  tracersX = zeros(2*N, nlayers);

  % Generate tracers
  h = vesicle.length/vesicle.N;  % arc-length spacing
  dlayer = [-h; -h/2; h/2; h];
  for il = 1 : nlayers
    tracersX(:,il) = [vesicle.X(1:end/2)+nx*dlayer(il);vesicle.X(end/2+1:end)+ny*dlayer(il)];
  end

  tracers.N = N;
  tracers.nv = nlayers;
  tracers.X = tracersX;

  G = op.stokesSLmatrix(vesicle);
  kernelDirect = @op.exactStokesSL;
  SLP = @(X) op.exactStokesSLdiag(vesicle,G,X);
  [~,NearV2T] = vesicle.getZone(tracers,2);
  
  % velocity on layers (the order is -h, -h/2, h/2, h)
  VelOnGridModesReal = zeros(2*N,nlayers,nmodes); 
  VelOnGridModesImag = zeros(2*N,nlayers,nmodes);
  % velocity on the layer on the vesicle
  selfVelModesReal = zeros(2*N,nmodes);
  selfVelModesImag = zeros(2*N,nmodes);
  for imode = 1 : nmodes
    forRealVels = [Br(:,imode); Bi(:,imode)];
    forImagVels = [-Bi(:,imode); Br(:,imode)];

    VelOnGridModesReal(:,:,imode) = op.nearSingInt(vesicle,forRealVels,SLP,NearV2T,kernelDirect,tracers,false);
    selfVelModesReal(:,imode) = G*forRealVels;
    

    VelOnGridModesImag(:,:,imode) = op.nearSingInt(vesicle,forImagVels,SLP,NearV2T,kernelDirect,tracers,false);
    selfVelModesImag(:,imode) = G*forImagVels;
  end
  fileName = ['./vesicleID_' num2str(ives) '.mat']; 
  save(fileName,'VelOnGridModesImag','VelOnGridModesReal','selfVelModesImag','selfVelModesReal','-v7.3')
  
 
end

end
