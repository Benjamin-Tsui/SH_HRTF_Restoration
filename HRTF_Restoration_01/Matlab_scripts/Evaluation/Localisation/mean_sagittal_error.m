function [mean_error_out,mean_error_in, m_out, m_in] = mean_sagittal_error(hrtf_out, hrtf_in, hrtf_ref, angle_matched)
%MEAN_SAGITTAL_ERROR Summary of this function goes here
%   Detailed explanation goes here

ele_idx = find(angle_matched(:,1) == 0);
ele = angle_matched(ele_idx(:), 2);
Fs = 48000;

% hrtf out to hrir out 
hrir_ele_tensor = zeros(512, 2, length(ele_idx));
hrir_ele_in_tensor = zeros(512, 2, length(ele_idx));
hrir_ref_tensor = zeros(512, 2, length(ele_idx));
for n = 1:length(ele_idx)
    hrtf_ele = [hrtf_out(ele_idx(n), 1:256)', hrtf_out(ele_idx(n), 257:512)'];
    %plot(hrtf);
    hrtf_ele =(10.^(hrtf_ele ./ 20));
    hrir_ele = ifft(hrtf_ele,512,1);
    hrir_ele = real(hrir_ele);
    hrir_ele = circshift(hrir_ele,256,1);
%     plot(hrir_ele) 
    hrir_ele_tensor(:,:,n) = hrir_ele;
    
    hrtf_ele_in = [hrtf_in(ele_idx(n), 1:256)', hrtf_in(ele_idx(n), 257:512)'];
    %plot(hrtf);
    hrtf_ele_in =(10.^(hrtf_ele_in ./ 20));
    hrir_ele_in = ifft(hrtf_ele_in,512,1);
    hrir_ele_in = real(hrir_ele_in);
    hrir_ele_in = circshift(hrir_ele_in,256,1);
%     plot(hrir_ele) 
    hrir_ele_in_tensor(:,:,n) = hrir_ele_in;

    hrtf_ref_temp = [hrtf_ref(ele_idx(n), 1:256)', hrtf_ref(ele_idx(n), 257:512)'];
    %plot(hrtf);
    hrtf_ref_temp =(10.^(hrtf_ref_temp ./ 20));
    hrir_ref = ifft(hrtf_ref_temp,512,1);
    hrir_ref = real(hrir_ref);
    hrir_ref = circshift(hrir_ref,256,1);
%     plot(hrir_ref)
    hrir_ref_tensor(:,:,n) = hrir_ref; 
end
hrir_ele_tensor = permute(hrir_ele_tensor,[1,3,2]); 
hrir_ele_in_tensor = permute(hrir_ele_in_tensor,[1,3,2]); 
hrir_ref_tensor = permute(hrir_ref_tensor,[1,3,2]); 

SOFA_FOR_TOM(permute(hrir_ele_tensor,[2,3,1]), zeros(length(ele),1), ele, 1.2, Fs, 'hrir_ele.sofa');
SOFA_FOR_TOM(permute(hrir_ele_in_tensor,[2,3,1]), zeros(length(ele),1), ele, 1.2, Fs, 'hrir_ele_in.sofa');
SOFA_FOR_TOM(permute(hrir_ref_tensor,[2,3,1]), zeros(length(ele),1), ele, 1.2, Fs, 'hrir_ref.sofa');

hrir_ele_tensor = SOFAload('hrir_ele.sofa');
hrir_ele_in_tensor = SOFAload('hrir_ele_in.sofa');
hrir_ref_tensor = SOFAload('hrir_ref.sofa');

[p_out,rang_out,tang_out] = baumgartner2014(hrir_ele_tensor, hrir_ref_tensor);
m_out = baumgartner2014_virtualexp(p_out,tang_out,rang_out);

mean_error_out = mean(abs(m_out(:,8) - m_out(:,6)));
fprintf(['mean sagittal error (model output):', num2str(mean_error_out) '\n'])

figure
colormap bone
plot_baumgartner2014( p_out,tang_out,rang_out,'cmax',0.0454);
title('model output')
colormap bone
% via expectancy values:
[qe_out,pe_out] = baumgartner2014_pmv2ppp(p_out,tang_out,rang_out,'print');
set(gcf, 'Color', 'w');

[p_in,rang_in,tang_in] = baumgartner2014(hrir_ele_in_tensor, hrir_ref_tensor);
m_in = baumgartner2014_virtualexp(p_in,tang_in,rang_in);

mean_error_in = mean(abs(m_in(:,8) - m_in(:,6)));
fprintf(['mean sagittal error (SH input):', num2str(mean_error_in) '\n'])

figure
colormap bone
plot_baumgartner2014( p_in,tang_in,rang_in,'cmax',0.0454);
title('SHed input')
colormap bone
% via expectancy values:
[qe_in,pe_in] = baumgartner2014_pmv2ppp(p_in,tang_in,rang_in,'print');
set(gcf, 'Color', 'w');

%[E,lat] = baumgartner2017( hrir_ele_in_tensor,hrir_ref_tensor );


end

