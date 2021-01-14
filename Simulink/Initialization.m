%% Variable Initialisations
A = [0.01   -0.4940 ; 0.1129 -0.3832];    % State Transition Matrix or Prediction Matrix
B = [0.5919 ; 0.5191];                    % Input Control Matrix
C = [1 0]; 
mn = 0.07;
pn = 0.09;
Ts = -1;
load('measurement_data.mat');
endtime = length(y_measured);