% %Load sparse HRIR dataset in SOFA format
% sparseHRIRdataset_SOFA = SOFAload('H19_48K_24bit_256tap_FIR_SOFA.sofa');
% %Transform to sparseHRTFdataset struct with pre-defined samplingGrid 
% %(Lebedev grid with 38 nodes here), Nmax = 4, and FFToversize = 4.
% input_array = supdeq_lebedev(6); 
% %'octo3' in iasxLDir.m
% input_array(:,1) = [90.0 270.0 0.0 0.0 180.0 180.0]';
% input_array(:,2) = [90.0   90.0 135.0 45.0 135.0 45.0];
%     
% sparseHRTFdataset = supdeq_sofa2hrtf_SADIE(sparseHRIRdataset_SOFA,1,input_array,4);
% 
% %% (3) - Get equalization dataset (SH-coefficients)
% %The eqDataset describes the sound pressure distribution on a sphere 
% %Use defaults: N = 35, earDistance = 0.165m, NFFT = 512, fs = 48000;
% eqDataset = supdeq_getEqDataset;
% 
% %% (4) - Perform equalization
% %Here, the sparse HRTF dataset is equalized with the eqDataset. The
% %equalized HRTF are transformed to the SH-domain again with the maximal 
% %order N which is possible with the sparse sampling grid.
% %N and the sparse sampling grid are part of the sparseHRTFdataset struct
% sparseSamplingGrid = sparseHRTFdataset.samplingGrid;
% Nsparse = sparseHRTFdataset.Nmax;
% 
% eqHRTFdataset = supdeq_eq(sparseHRTFdataset,eqDataset,Nsparse,sparseSamplingGrid);
% 
% %% (5) - Perform de-equalization 
% %Here, the sparse equalized HRTF dataset is de-equalized with the
% %deqDataset. This is done on a dense spatial sampling grid. The results is a
% %dense HRTF/HRIR dataset. In this case, deqDataset is the same as the
% %eqDataset...
% 
% %First, define dense spatial sampling grid. Here, we use the lebedev grid
% %with 2702 points again (same as the reference HRIR dataset).
% %The highest stable grid order here is N = 44. However, we use N = 35 for the
% %spherical Fourier transform of the de-equalized HRTFs.
% % denseSamplingGrid = supdeq_lebedev(2702);
% % Ndense = 35;
% 
% denseSamplingGrid = sparseHRIRdataset_SOFA.SourcePosition(:,1:2);
% denseSamplingGrid(:,2) = denseSamplingGrid(:,2) + 90;
% Ndense = 35;
% 
% %Perform de-equalization. Apply head and tail window (8 and 32 samples
% %respectively) to de-equalized HRIRs/HRTFs.
% [denseHRTFdataset, denseHRIRdataset, denseHRTFdataset_sh] = supdeq_deq(eqHRTFdataset, eqDataset, Ndense, denseSamplingGrid,[8,32]);
% 
% %% (6) - Optional: Save as SOFA object
% %Use defaults: fs = 48000, earDistance = 0.165m, sourceDistance = 3.0m
% denseHRIRdataset_SOFA = supdeq_writeSOFAobj(denseHRIRdataset.HRIR_L,denseHRIRdataset.HRIR_R,denseSamplingGrid);
% 
% nfft = 512;
% 
% hrir_out_left = squeeze(denseHRIRdataset_SOFA.Data.IR(:,1,:));
% hrir_out_right = squeeze(denseHRIRdataset_SOFA.Data.IR(:,2,:));
% % hrir to hrtf
% hrtf_out_left = abs(fft(hrir_out_left', nfft))';
% hrtf_out_right = abs(fft(hrir_out_right', nfft))';
% % convert to dB
% hrtf_out_dB_left = 20*log10(abs(hrtf_out_left));
% hrtf_out_dB_right = 20*log10(abs(hrtf_out_right));
% 
% % remove mirror part of the result
% hrtf_out_left = hrtf_out_left(:, 1: size(hrtf_out_left,2)/2);
% hrtf_out_right = hrtf_out_right(:, 1: size(hrtf_out_right,2)/2);
% hrtf_out_dB_left = hrtf_out_dB_left(:, 1: size(hrtf_out_dB_left,2)/2);
% hrtf_out_dB_right = hrtf_out_dB_right(:,1: size(hrtf_out_dB_right,2)/2);
% 
% supdeq_hrtf = [hrtf_out_dB_left hrtf_out_dB_right];
% 
% out_angle = denseHRIRdataset_SOFA.SourcePosition(:,1:2);
% out_angle(:,1) = abs(out_angle(:,1) .* -1);
% out_angle(out_angle(:,1)>180,1) = out_angle(out_angle(:,1)>180,1) - 360;
% out_angle(:,2) = out_angle(:,2) .* -1;
% 
% input_hrtf = readmatrix(...
%     'ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/SHed_hrtf_dB.txt');
% % change result
% predict_hrtf = readmatrix('SH_reconstruct/model_ouput/training_HRTF_08_08_sparse_02_sub20_out.csv');
% target_hrtf = readmatrix(...
%     'ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/hrtf_dB.txt');
% angle = readmatrix(...
%     'ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/angles.txt');
% 
% [~,~,angle_idx] = intersect(round(angle(:,1:2),1),round(out_angle,1),'rows','stable');
% supdeq_hrtf = supdeq_hrtf(angle_idx,:);

%% Code above only required to run once for initialisation

idx = 1550;

hrtf_fs = 48000;
hrir_length = 512;

cn = dsp.ColoredNoise(1,48000,2,'Color','pink','SamplesPerFrame',480000);
x = cn();
audio_in = x ./ max(max(x)) * 0.9;
% [audio_in,Fs] = audioread(input_file);

% add back mirror part in hrtf
input_hrtf_l = [input_hrtf(:, 1:256) fliplr(input_hrtf(:, 1:256))];
input_hrtf_r = [input_hrtf(:, 257:512) fliplr(input_hrtf(:, 257:512))];
predict_hrtf_l = [predict_hrtf(:, 1:256) fliplr(predict_hrtf(:, 1:256))];
predict_hrtf_r = [predict_hrtf(:, 257:512) fliplr(predict_hrtf(:, 257:512))];
target_hrtf_l = [target_hrtf(:, 1:256) fliplr(target_hrtf(:, 1:256))];
target_hrtf_r = [target_hrtf(:, 257:512) fliplr(target_hrtf(:, 257:512))];
supdeq_hrtf_l = [supdeq_hrtf(:, 1:256) fliplr(supdeq_hrtf(:, 1:256))];
supdeq_hrtf_r = [supdeq_hrtf(:, 257:512) fliplr(supdeq_hrtf(:, 257:512))];

% convert from dB
input_hrtf_l = 10.^(input_hrtf_l ./20); 
input_hrtf_r = 10.^(input_hrtf_r ./20); 
predict_hrtf_l = 10.^(predict_hrtf_l ./20); 
predict_hrtf_r = 10.^(predict_hrtf_r ./20); 
target_hrtf_l = 10.^(target_hrtf_l ./20);  
target_hrtf_r = 10.^(target_hrtf_r ./20);  
supdeq_hrtf_l = 10.^(supdeq_hrtf_l ./20);  
supdeq_hrtf_r = 10.^(supdeq_hrtf_r ./20);  

input_hrir_l = circshift(real(ifft(input_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
input_hrir_r = circshift(real(ifft(input_hrtf_r, hrir_length, 2)), hrir_length/2, 2);
predict_hrir_l = circshift(real(ifft(predict_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
predict_hrir_r = circshift(real(ifft(predict_hrtf_r, hrir_length, 2)), hrir_length/2, 2);
target_hrir_l = circshift(real(ifft(target_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
target_hrir_r = circshift(real(ifft(target_hrtf_r, hrir_length, 2)), hrir_length/2, 2);
supdeq_hrir_l = circshift(real(ifft(supdeq_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
supdeq_hrir_r = circshift(real(ifft(supdeq_hrtf_r, hrir_length, 2)), hrir_length/2, 2);

out_input_l = conv(audio_in(:,1)', input_hrir_l(idx,:));
out_input_r = conv(audio_in(:,2)', input_hrir_r(idx,:));
out_input = [out_input_l', out_input_r'];
out_input = out_input ./ max(max(out_input)) * 0.9;
out_predict_l = conv(audio_in(:,1)', predict_hrir_l(idx,:));
out_predict_r = conv(audio_in(:,2)', predict_hrir_r(idx,:));
out_predict = [out_predict_l', out_predict_r'];
out_predict = out_predict ./ max(max(out_predict)) * 0.9;
out_target_l = conv(audio_in(:,1)', target_hrir_l(idx,:));
out_target_r = conv(audio_in(:,2)', target_hrir_r(idx,:));
out_target = [out_target_l', out_target_r'];
out_target = out_target ./ max(max(out_target)) * 0.9;
out_supdeq_l = conv(audio_in(:,1)', supdeq_hrir_l(idx,:));
out_supdeq_r = conv(audio_in(:,2)', supdeq_hrir_r(idx,:));
out_supdeq = [out_supdeq_l', out_supdeq_r'];
out_supdeq = out_supdeq ./ max(max(out_supdeq)) * 0.9;


% sound(out_input, 48000);
% sound(out_predict, 48000);
% sound(out_target, 48000);
% sound(out_supdeq, 48000);

audiowrite('SH_reconstruct/input.wav',out_input,48000);
audiowrite('SH_reconstruct/predict.wav',out_predict,48000);
audiowrite('SH_reconstruct/target.wav',out_target,48000);
audiowrite('SH_reconstruct/supdeq.wav',out_supdeq,48000);


