function [output_hrir_left, output_hrir_right] = time_align_ambisonic_HRIRs(hrir_left, hrir_right, Fs,ambiOrder)

% ------------------------------------------------------------------------
% function - time_align_ambisonic_HRIRs()
% time aligns hrirs above f_alias, unless ambiOrder is less than 4 in which
% case time aligns above f_alias of m=4.
% ---------------
%
% inputHRIR             - matrix of stereo impulse responses (in format
% [num of channels x length of impulse responses x num of impulse
% responses]
%
% stretchFactor         - single number for amount of stretch
%
% Fs                    - sample rate in hz
%
% ambiOrder             - order of Ambisonics (which determines xover freq)
%
% DecodeLoudspeakerDirections - for diffuse-field equalisation calculation
%
%
%
% ------------------------------------------------------------------------
% Thomas McKenzie, University of York, 20 / 4 / 2018

inputHRIR = cat(3, hrir_left, hrir_right);
inputHRIR = permute(inputHRIR,[3 2 1]);


% generate low-pass and high-pass crossover filters ----- IF lower than
% 4th order, only want to cross over to time aligned at ~2.5khz (so use crossover filter for 4th order).
if ambiOrder < 4
    [filtLo,filtHi] = ambisonicCrossoverFilter_Tom(4,Fs);
else
    [filtLo,filtHi] = ambisonicCrossoverFilter_Tom(ambiOrder,Fs);
end

inputHRIR_padded = cat(2, inputHRIR, zeros(size(inputHRIR, 1), ceil(size(filtLo, 2)), size(inputHRIR, 3)));

% remove ITDs / time align
inputHRIR_exaggerated = removeITD(inputHRIR_padded);

% crossover
for i = 1 : length(inputHRIR_padded(1,1,:))
    for j = 1:length(inputHRIR_padded(:,1,1))
        outputFilt_low(j,:,i) = filter(filtLo,1,inputHRIR_padded(j,:,i));
        outputFilt_High(j,:,i) = filter(filtHi,1,inputHRIR_exaggerated(j,:,i));
    end
end

% Add the low freq and high freq together
outputHRIRs = (outputFilt_low + outputFilt_High);

% filter delay compensation
gd = mean(grpdelay(filtLo)) ;
output_hrir_left = squeeze(outputHRIRs(1, :, :))';
output_hrir_right = squeeze(outputHRIRs(2, :, :))';
output_hrir_left = output_hrir_left(:, gd+1: gd+ size(inputHRIR, 2));
output_hrir_right = output_hrir_right(:, gd+1: gd+ size(inputHRIR, 2));

% Tom's original implementation 
% % remove ITDs / time align
% inputHRIR_exaggerated = removeITD(inputHRIR);
% % crossover
% for i = 1 : length(inputHRIR(1,1,:))
%     for j = 1:length(inputHRIR(:,1,1))
%         outputFilt_low_tom(j,:,i) = filter(filtLo,1,inputHRIR(j,:,i));
%         outputFilt_High_tom(j,:,i) = filter(filtHi,1,inputHRIR_exaggerated(j,:,i));
%     end
% end
% 
% % Add the low freq and high freq together
% outputHRIRs = (outputFilt_low_tom + outputFilt_High_tom);
% 
% for i = 1 : length(inputHRIR(:,1,1))
%     for j = 1:length(inputHRIR(1,1,:))
%         outputHRIRs(i,:,j) = delayseq(outputHRIRs(i,:,j)', -gd);
%     end
% end

disp('Time alignment complete!');

end
