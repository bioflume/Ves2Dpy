classdef MLARM_py

  properties
  dt
  vinf
  oc
  advNetInputNorm
  advNetOutputNorm
  relaxNetInputNorm
  relaxNetOutputNorm
  area0 % initial area of vesicle
  len0 % initial length of vesicle
  end

  methods
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function o = MLARM_py(dt,vinf,oc,advNetInputNorm,advNetOutputNorm,relaxNetInputNorm,relaxNetOutputNorm)
  o.dt = dt; % time step size
  o.vinf = vinf; % background flow (analytic -- input as function of vesicle config)
  o.oc = oc; % curve class

  % Normalization values for advection (translation) networks
  o.advNetInputNorm = advNetInputNorm;
  o.advNetOutputNorm = advNetOutputNorm;

  % Normalization values for relaxation network
  o.relaxNetInputNorm = relaxNetInputNorm;
  o.relaxNetOutputNorm = relaxNetOutputNorm;

  end % MLARM_py
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function Xnew = time_step(o,X)
  % take a time step with neural networks
  oc = o.oc;
  vback = o.vinf(X);

  % 1) Translate vesicle with network
  Xadv = o.translateVinfNet(X,vback);

  % Correct area and length
  XadvC = oc.correctAreaAndLength(Xadv, o.area0, o.len0);
  Xadv = oc.alignCenterAngle(Xadv, XadvC);

  % 2) Relax vesicle with network
  Xnew = o.relaxNet(Xadv);

  % Correct area and length
  XnewC = oc.correctAreaAndLength(Xnew, o.area0, o.len0);
  Xnew = oc.alignCenterAngle(Xnew, XnewC);
  end % time_step
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function Xadv = translateVinfNet(o,X,vback)
  % translate vesicle using networks

  % Standardize vesicle (zero center, pi/2 inclination angle, equil dist)
  [Xstand, scaling, rotate, trans, sortIdx] = o.standardizationStep(X);

  % Normalize input
  input_net = zeros(nv, 2, 128);
  for imode = 2 : 128
    % mean and std for the input to imode_th network
    x_mean = o.advNetInputNorm(imode-1,1); 
    x_std = o.advNetInputNorm(imode-1,2);
    y_mean = o.advNetInputNorm(imode-1,3);
    y_std = o.advNetInputNorm(imode-1,4); 

    input_net(:,1,:) = (Xstand(1:end/2)-x_mean)/x_std;
    input_net(:,2,:) = (Xstand(end/2+1:end)-y_mean)/y_std;
  end

  % Predict using neural networks (for MATLAB to use PyTorch we have
  % interface-specific codes. This should be easier in Python)
  input_conv = py.numpy.array(input_net);
  [Xpredict] = pyrunfile("advect_predict_pro.py","output_list",input_shape=input_conv,num_ves=py.int(nv));

  % Above line approximates multiplication M*(FFTBasis) 
  % Now, reconstruct Mvinf = (M*FFTBasis) * vinf_hat
  Z11 = zeros(128,128); Z12 = Z11; Z21 = Z11; Z22 = Z11;

  for imode = 2 : 128 % the first mode is zero
    % call every mode's output
    pred = double(Xpredict{imode-1}); % size(pred) = [1 2 256]
    
    % denormalize output
    real_mean = o.advNetOutputNorm(imode-1,1);
    real_std = o.advNetOutputNorm(imode-1,2);
    imag_mean = o.advNetOutputNorm(imode-1,3);
    imag_std = o.advNetOutputNorm(imode-1,4);

    % first channel is real
    pred(1,1,:) = (pred(1,1,:)*real_std) + real_mean;
    % second channel is imaginary
    pred(1,2,:) = (pred(1,2,:)*imag_std) + imag_mean;

    Z11(:,imode,1) = pred(1,1,1:end/2);
    Z21(:,imode,1) = pred(1,1,end/2+1:end);
    Z12(:,imode,1) = pred(1,2,1:end/2);
    Z22(:,imode,1) = pred(1,2,end/2+1:end);

  end % imode

  % Take fft of the velocity (should be standardized velocity)
  % only sort points and rotate to pi/2 (no translation, no scaling)
  vinfStand = o.standardize(vback,[0;0],rotate,1,sortIdx);
  z = vinfStand(1:end/2)+1i*vinfStand(end/2+1:end);

  zh = fft(z);
  V1 = real(zh); V2 = imag(zh);
  % Compute the approximate value of the term M*vinf
  MVinf = [Z11*V1+Z12*V2; Z21*V1+Z22*V2];
  
  % update the standardized shape
  XnewStand = o.dt*vinfStand - o.dt*MVinf;

  % destandardize
  Xadv = o.destandardize(XnewStand,trans,rotate,scaling,sortIdx);

  % add the initial since solving dX/dt = (I-M)vinf
  Xadv = X + Xadv;

  end % translateVinfNet
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function Xnew = relaxNet(o,X)
  % Standardize vesicle
  [Xin, scaling, rotate, trans, sortIdx] = o.standardizationStep(X);

  % Normalize input
  x_mean = o.relaxNetInputNorm(1);
  x_std = o.relaxNetInputNorm(2);
  y_mean = o.relaxNetInputNorm(3);
  y_std = o.relaxNetInputNorm(4);

  Xin(1:end/2) = (Xin(1:end/2)-x_mean)/x_std;
  Xin(end/2+1:end) = (Xin(end/2+1:end)-y_mean)/y_std;

  % Adjust the input shape for the network
  XinitShape = zeros(1,2,128);
  XinitShape(1,1,:) = Xin(1:end/2)'; 
  XinitShape(1,2,:) = Xin(end/2+1:end)';
  XinitConv = py.numpy.array(XinitShape);

  % Make prediction -- needs to be adjusted for python
  [XpredictStand] = pyrunfile("relax_predict_dt1E5_pro.py", "predicted_shape", input_shape=XinitConv);
  
   % Normalize output
  x_mean = o.relaxNetOutputNorm(1);
  x_std = o.relaxNetOutputNorm(2);
  y_mean = o.relaxNetOutputNorm(3);
  y_std = o.relaxNetOutputNorm(4);

  % Denormalize output
  Xpred = zeros(size(Xin));
  XpredictStand = double(XpredictStand);

  Xpred(1:end/2) = XpredictStand(1,1,:)*x_std + x_mean;
  Xpred(end/2+1:end) = XpredictStand(1,2,:)*y_std + y_mean;

  % destandardize
  Xnew = o.destandardize(Xpred,trans,rotate,scaling,sortIdx);

  end % relaxNet
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function [X,scaling,rotate,trans,sortIdx] = standardizationStep(o,Xin)
  oc = o.oc;
  X = Xin;
  % Equally distribute points in arc-length
  for iter = 1 : 5
    [X,~,~] = oc.redistributeArcLength(X);
  end
  % Fix misalignment in center and angle due to reparametrization
  X = oc.alignCenterAngle(Xin,X);

  % standardize angle, center, scaling and point order
  [trans,rotate,scaling,sortIdx] = o.referenceValues(X);
  X = o.standardize(X,trans,rotate,scaling,sortIdx);
  end % standardizationStep

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function XrotSort = standardize(o,X,translation,rotation,scaling,sortIdx)
  N = numel(sortIdx);

  % translate, rotate and scale configuration
  Xrotated = scaling*o.rotationOperator(o.translateOp(X,translation),rotation);   

  % now order the points
  XrotSort = [Xrotated(sortIdx);Xrotated(sortIdx+N)];

  end % standardize
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function X = destandardize(o,XrotSort,translation,rotation,scaling,sortIdx)

  N = numel(sortIdx);    
    
  % change ordering back 
  X = zeros(size(XrotSort));
  X([sortIdx;sortIdx+N]) = XrotSort;

  % scaling back
  X = X/scaling;

  % take rotation back
  cx = mean(X(1:end/2)); cy = mean(X(end/2+1:end));
  X = o.rotationOperator([X(1:end/2)-cx;X(end/2+1:end)-cy],-rotation);
  X = [X(1:end/2)+cx; X(end/2+1:end)+cy];

  % take translation back
  X = o.translateOp(X,-translation);

  end % destandardize

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function [translation,rotation,scaling,sortIdx] = referenceValues(o,Xref)
  oc = o.oc;
  N = numel(Xref)/2;

  % find translation, rotation and scaling
  translation = [-mean(Xref(1:end/2));-mean(Xref(end/2+1:end))];
  rotation = pi/2-oc.getIncAngle2(Xref);
    
  % amount of scaling
  [~,~,length] = oc.geomProp(Xref);
  scaling = 1/length;
    
  % find the ordering of the points
  Xref = scaling*o.rotationOperator(o.translateOp(Xref,translation),rotation);

  firstQuad = find(Xref(1:end/2)>=0 & Xref(end/2+1:end)>=0);
  theta = atan2(Xref(end/2+1:end),Xref(1:end/2));
  [~,idx]= min(theta(firstQuad));
  sortIdx = [(firstQuad(idx):N)';(1:firstQuad(idx)-1)'];

  end % referenceValues
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function Xrot = rotationOperator(o,X,theta)
  % Get x-y coordinates
  Xrot = zeros(size(X));
  x = X(1:end/2); y = X(end/2+1:end);

  % Rotated shape
  xrot = (x)*cos(theta) - (y)*sin(theta);
  yrot = (x)*sin(theta) + (y)*cos(theta);

  Xrot(1:end/2) = xrot;
  Xrot(end/2+1:end) = yrot;
  end % rotationOperator
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  function Xnew = translateOp(o,X,transXY)
  Xnew = zeros(size(X));
  Xnew(1:end/2) = X(1:end/2)+transXY(1);
  Xnew(end/2+1:end) = X(end/2+1:end)+transXY(2);
  end  % translateOp  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end % methods

end % MLARM_py
