function [filtLo,filtHi,fcHz] = ambisonicCrossoverFilter_Tom(ambiOrderOrfcHz,Fs)
% 
% Creates the low and high pass crossover filters for ambisonic dual-band
% rendering. 
% ======================================
% Inputs: ambiOrder, Fs
% 
% Outputs: filtLo, filtHi
% 
% example for using filters:
% filteredHRIR(j,:,i) = filter(filtLo,1,inputHRIR(j,:,i));
% 
% ------------------------------------------------------------------------
% initial version 20 / 6 / 2018
% updated version 1 / 11 / 2018 changed fc1 calculation to equation (2) from \cite{Bertet2013}
% ------------------------------------------------------------------------
% -------- Thomas McKenzie, University of York, 1 / 11 / 2018 ------------
% ------------------------------------------------------------------------

if ambiOrderOrfcHz <= 36 % if the first argument is a value of 36 or smaller, it is assumed the input is the ambisonic order. Otherwise, it is assumed the input is the cut off frequency. 
    R = 0.09; % radius of reproduction area in metres (Neumann KU 100 is stated on the website as having a width of 18cm)
    % fc1 = (ambiOrder * 600) + 100; % calculated from Table 1 in \cite{Bertet2013}.
    fcHz = (ambiOrderOrfcHz * 343) / (4*R*(ambiOrderOrfcHz + 1) * sin(pi / (2*ambiOrderOrfcHz+2))); % this is equation (2) from \cite{Bertet2013}
else
    fcHz = ambiOrderOrfcHz;
end

fcNorm = fcHz/(Fs/2); % convert to normalised frequency

XoverOrder = 128;%/4; % crossover order ----- if big lengths of impulse response should be 128, but if small (like sadie 2) then should be 128/4
ripple1 = 50; % ripple in dB

% using chebyshev windows
filtLo = fir1(XoverOrder,fcNorm,'low',chebwin((XoverOrder+1),ripple1), 'noscale');
filtHi = fir1(XoverOrder,fcNorm,'high',chebwin((XoverOrder+1),ripple1), 'noscale');

end

