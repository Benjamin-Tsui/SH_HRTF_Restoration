
input_hrtf = readmatrix(...
    'ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/SHed_hrtf_dB.txt');
% change result
predict_hrtf = readmatrix('SH_reconstruct/model_ouput/training_HRTF_08_08_sparse_02_sub20_out.csv');
target_hrtf = readmatrix(...
    'ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/hrtf_dB.txt');
angle = readmatrix(...
    'ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_sub_20_oct_3/angles.txt');
idx = 29;

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

% convert from dB
input_hrtf_l = 10.^(input_hrtf_l ./20); 
input_hrtf_r = 10.^(input_hrtf_r ./20); 
predict_hrtf_l = 10.^(predict_hrtf_l ./20); 
predict_hrtf_r = 10.^(predict_hrtf_r ./20); 
target_hrtf_l = 10.^(target_hrtf_l ./20);  
target_hrtf_r = 10.^(target_hrtf_r ./20);  

input_hrir_l = circshift(real(ifft(input_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
input_hrir_r = circshift(real(ifft(input_hrtf_r, hrir_length, 2)), hrir_length/2, 2);
predict_hrir_l = circshift(real(ifft(predict_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
predict_hrir_r = circshift(real(ifft(predict_hrtf_r, hrir_length, 2)), hrir_length/2, 2);
target_hrir_l = circshift(real(ifft(target_hrtf_l, hrir_length, 2)), hrir_length/2, 2);
target_hrir_r = circshift(real(ifft(target_hrtf_r, hrir_length, 2)), hrir_length/2, 2);

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

% sound(out_input, 48000);
% sound(out_predict, 48000);
% sound(out_target, 48000);
% 
% audiowrite('SH_reconstruct/input.wav',out_input,48000);
% audiowrite('SH_reconstruct/predict.wav',out_predict,48000);
% audiowrite('SH_reconstruct/target.wav',out_target,48000);


