function [inputHRIR_ITDremoved] = removeITD(inputHRIR)

% ------------------------------------------------------------------------
% function - exaggerateITD()
% A function to increase ITDs in high-frequencies of HRIRs for Ambisonic
% rendering.
% ---------------
%
% inputHRIR             - matrix of stereo impulse responses (in format
% [num of channels x length of impulse responses x num of impulse
% responses]
%
%
% ------------------------------------------------------------------------
% Thomas McKenzie, University of York, 20 / 4 / 2018

inputHRIR_ITDremoved = zeros(length(inputHRIR(:,1,1)),length(inputHRIR(1,:,1)),length(inputHRIR(1,1,:)));

FcNorm = 500/(48000/2); % convert cutoff freq to normalised freq
XoverOrder = 8; % VERY BIG order to make sharp filter. 
filtLo = fir1(XoverOrder,FcNorm,'low',hamming((XoverOrder+1)), 'noscale');


% remove ITD of each HRIR compared to the first
for i = 1 : length(inputHRIR(1,1,:))
    for j = 1:length(inputHRIR(:,1,1))

hrir_1_f = filter(filtLo,1,inputHRIR(1,:,1));
hrir_i_f = filter(filtLo,1,inputHRIR(j,:,i));
hrirDelay = finddelay(hrir_1_f,hrir_i_f);

%         hrirDelay = finddelay(inputHRIR(1,:,1),inputHRIR(j,:,i));
        
        if hrirDelay>=0
            inputHRIR_ITDremoved(j,:,i) = [inputHRIR(j,1+hrirDelay:length(inputHRIR(j,:,i)),i), zeros(1,hrirDelay,1)];
        elseif hrirDelay < 0
            inputHRIR_ITDremoved(j,:,i) = [zeros(1,abs(hrirDelay),1),inputHRIR(j,1:length(inputHRIR(j,:,i))-abs(hrirDelay),i)];
        end
        
    end
end

% Then remove all shared ITD:
inputHRIR1 = mean(mean(inputHRIR,1),3);
% inputHRIR1 = inputHRIR(1,:,1);
inputHRIR2 = mean(mean(inputHRIR_ITDremoved,1),3);


% hrir_1_f = filter(filtLo,1,inputHRIR1);
% hrir_i_f = filter(filtLo,1,inputHRIR2);
% overallDelay = finddelay(hrir_1_f,hrir_i_f);

overallDelay = finddelay(inputHRIR1,inputHRIR2);

if overallDelay>=0
    inputHRIR_ITDremoved = [inputHRIR_ITDremoved(:,1+overallDelay:length(inputHRIR_ITDremoved(1,:,1)),:), zeros(length(inputHRIR(:,1,1)),overallDelay,length(inputHRIR(1,1,:)))];
elseif overallDelay < 0
%     inputHRIR_ITDremoved(j,:,i) = [zeros(length(inputHRIR(:,1,1)),abs(overallDelay),length(inputHRIR(1,1,:))),inputHRIR_ITDremoved(:,1:length(inputHRIR_ITDremoved(1,:,1))-abs(overallDelay),:)];
    inputHRIR_ITDremoved = [zeros(length(inputHRIR(:,1,1)),abs(overallDelay),length(inputHRIR(1,1,:))),inputHRIR_ITDremoved(:,1:length(inputHRIR_ITDremoved(1,:,1))-abs(overallDelay),:)];
end

end
