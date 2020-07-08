function [PSD_in_summary,PSD_out_summary, PSD_in_angle, PSD_out_angle] = compare_PSD_Bern(model, plotflag)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% plot flag:
% 0 = no plot
% 1 = 3d plot
% 2 = heatmap
% 3 = both 3d plot and heat map 

% example:
% model = 'training_HRTF_08++_12_sparse'; % the name of the saved model
% [PSD_in_bern_summary,PSD_out_bern_summary, PSD_in_bern, PSD_out_bern] = compare_PSD_Bern(model,1);

%% csv results 
% change csv here 
hrtf_out = readmatrix(['SH_reconstruct/model_ouput/' model '_bern_out.csv']);
hrtf_in = readmatrix('ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/SHed_hrtf_dB.txt');
hrtf_tar = readmatrix('ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/hrtf_dB.txt');
angle_matched = readmatrix('ML_HRTF_Data/Time_aligned/SH_HRTFs_1st_order_512_sparse_in_bern_oct_3/angles.txt');

% hrtf_out = add_back_low_freq(hrtf_out, hrtf_in);

hrtf_dB_left_in = hrtf_in(:, 1:256);
hrtf_dB_right_in = hrtf_in(:, 257:512);
hrtf_dB_left_out = hrtf_out(:, 1:256);
hrtf_dB_right_out = hrtf_out(:, 257:512);
hrtf_dB_left_tar = hrtf_tar(:, 1:256);
hrtf_dB_right_tar = hrtf_tar(:, 257:512);

fs = 48000;
nfft = 512;

%% compare angle
in = cat(3,hrtf_dB_left_in,hrtf_dB_right_in);
out = cat(3,hrtf_dB_left_out,hrtf_dB_right_out);
target = cat(3,hrtf_dB_left_tar,hrtf_dB_right_tar); 

in = permute(in,[2 1 3]);
out = permute(out,[2 1 3]);
target = permute(target,[2 1 3]);

s = GetVoronoiPlotandSolidAng(angle_matched(:,1)', angle_matched(:,2)', 0);
s_check = sum(sum(s)); % should be 1
freq = 0 : fs/nfft : fs-(fs/nfft);
freq = freq(1:end/2);
limit = 1;
offset_F = 'none';

%% compare and consolidate to one number
clear specDiffAmbiSAW avgDiff_ERB 
[diff, avgDiff_ERB, offset_F] = PspecComp_10(in, target, s, freq, limit, offset_F);

for i = 1: length(avgDiff_ERB(1,:,1))
    specDiffAmbiSAW(i,:) = avgDiff_ERB(:,i,:) * s(i);
end

weightedAmbiSpecDiffL = sum(specDiffAmbiSAW(:,1));
weightedAmbiSpecDiffR = sum(specDiffAmbiSAW(:,2));
weightedAmbiSpecDiffLandR = ((weightedAmbiSpecDiffL+weightedAmbiSpecDiffR)/2);

disp('Bernschutz')

PSD_mean = weightedAmbiSpecDiffLandR;
disp(strcat('mean PSD (input) = ',num2str(PSD_mean)))

PSD_angle = ((avgDiff_ERB(1,:,1)+avgDiff_ERB(1,:,2))/2);
disp(strcat('min PSD (input) = ',num2str(min(PSD_angle))));
disp(strcat('max PSD (input) = ',num2str(max(PSD_angle))));
disp(strcat('PSD range(input) = ',num2str(max(PSD_angle) - min(PSD_angle))));
PSD_in_summary = [max(PSD_angle) ; PSD_mean ; min(PSD_angle)];
PSD_in_angle = PSD_angle;

%% Graphic

if plotflag == 1 || plotflag == 3
    
    % For plotting - make maximum grayscale colour 1
    %     s1 = s ./ 0.0976;
    %     s1 = scaledata(s,0,1);
    
    
    hold off
    f = figure;%'rend','painters','pos',[10 10 700 430]);
    clf(f);
    set(f,'Renderer','zbuffer');
    ax = axes('Parent', f);
    hold(ax, 'on');
    axis(ax,'equal');
    
    [x, y, z] = sph2cart(deg2rad(angle_matched(:,1)'), deg2rad(angle_matched(:,2)'), ones(size(angle_matched(:,1)')));

    xyz = [x; y; z];
    xyz = bsxfun(@rdivide, xyz, sqrt(sum(xyz.^2,1)));
    [P, K, voronoiboundary] = voronoisphere(xyz);
    
    plot3(ax, xyz(1,:),xyz(2,:),xyz(3,:),'w.');
    
    n = length(PSD_angle);
    for k = 1:n
        X = voronoiboundary{k};
        %                 cl = [(s1(k)*0.99),(0.2 - (s1(k)*0.2)),(1 - (s1(k)))]; % smallest solid angles in blue, biggest in red
        %         cl = [1 - s1(k),1 - s1(k),1 - s1(k)]; % smallest solid angles in white, biggest in black
        
        
        for i = 1:length(X(:,1))
            for j = 1:length(X(1,:))
                if isnan(X(i,j))
                    X(i,j) = 0;
%                     ere = 1
                end
            end
        end
        
        
        fill3(X(1,:),X(2,:),X(3,:),PSD_angle(k),'Parent',ax,'EdgeColor','w'); % for white outline / edge colour
%         fill3(X(1,:),X(2,:),X(3,:),s(k),'Parent',ax,'EdgeColor','none');     % for no edge colour / outline
    end
    

    c2 = colorbar;
    c2.Label.String = 'PSD';% (x 10^{-2})';
    
    view([45 18])

    
    xlabel ( 'X axis' )
    ylabel ( 'Y axis' )
    zlabel ( 'Z axis' )
    
    grid on
    grid minor
    ax.MinorGridAlphaMode = 'manual';
    ax.MinorGridAlpha = 0.1;
    ax.FontSize = 14;
    
    colormap(flipud(parula));
    
    axis(ax,'equal');
    axis(ax,[-1 1 -1 1 -1 1]);
    
    
    xticks([-1 0 1])
    yticks([-1 0 1])
    zticks([-1 0 1])

    set(gcf, 'Color', 'w');
    
    dcm_obj = datacursormode;
    set(dcm_obj,'UpdateFcn',@disp_ploar)
    % display spherical coordinates (in degrees) instead of cartesian coordinates
    
    title(['Bernschutz (input), mean PSD = ' num2str(PSD_mean)])
    
    top = max(PSD_angle);
    caxis([0 top]);
    rotate3d on
end

if plotflag == 2 || plotflag == 3
    figure
    heatmap_plot(angle_matched(:,1)',angle_matched(:,2)',PSD_angle)
    title(['\fontsize{21}Bernschutz (input), mean PSD = ' num2str(PSD_mean)])
    xlabel('Azimuth (°)','FontSize',18)
    ylabel('Elevation (°)','FontSize',18)
    colormap(flipud(parula));
    c2 = colorbar;
    c2.Label.String = 'PSD';% (x 10^{-2})';
    top = max(PSD_angle);
    caxis([0 top]);
    grid on
    grid minor
    set(gcf, 'Position',  [100, 100, 900, 500])
end

%% compare angle
in = cat(3,hrtf_dB_left_in,hrtf_dB_right_in);
out = cat(3,hrtf_dB_left_out,hrtf_dB_right_out);
target = cat(3,hrtf_dB_left_tar,hrtf_dB_right_tar); 

in = permute(in,[2 1 3]);
out = permute(out,[2 1 3]);
target = permute(target,[2 1 3]);

s = GetVoronoiPlotandSolidAng(angle_matched(:,1)', angle_matched(:,2)', 0);
s_check = sum(sum(s)); % should be 1
freq = 0 : fs/nfft : fs-(fs/nfft);
freq = freq(1:end/2);
limit = 1;
offset_F = 'none';

%% compare and consolidate to one number
clear specDiffAmbiSAW avgDiff_ERB 

[diff, avgDiff_ERB, offset_F] = PspecComp_10(out, target, s, freq, limit, offset_F);

for i = 1: length(avgDiff_ERB(1,:,1))
    specDiffAmbiSAW(i,:) = avgDiff_ERB(:,i,:) * s(i);
end

weightedAmbiSpecDiffL = sum(specDiffAmbiSAW(:,1));
weightedAmbiSpecDiffR = sum(specDiffAmbiSAW(:,2));
weightedAmbiSpecDiffLandR = ((weightedAmbiSpecDiffL+weightedAmbiSpecDiffR)/2);

PSD_mean = weightedAmbiSpecDiffLandR;
disp(strcat('mean PSD (reconstructed) = ',num2str(PSD_mean)))

PSD_angle = ((avgDiff_ERB(1,:,1)+avgDiff_ERB(1,:,2))/2);
disp(strcat('min PSD (reconstructed) = ',num2str(min(PSD_angle))));
disp(strcat('max PSD (reconstructed) = ',num2str(max(PSD_angle))));
disp(strcat('PSD range(reconstructed) = ',num2str(max(PSD_angle) - min(PSD_angle))));
PSD_out_summary = [max(PSD_angle) ; PSD_mean ; min(PSD_angle)];
PSD_out_angle = PSD_angle;

%% Graphic

if plotflag == 1 || plotflag == 3
    
    % For plotting - make maximum grayscale colour 1
    %     s1 = s ./ 0.0976;
    %     s1 = scaledata(s,0,1);
    
    
    hold off
    f = figure;%'rend','painters','pos',[10 10 700 430]);
    clf(f);
    set(f,'Renderer','zbuffer');
    ax = axes('Parent', f);
    hold(ax, 'on');
    axis(ax,'equal');
    
    [x, y, z] = sph2cart(deg2rad(angle_matched(:,1)'), deg2rad(angle_matched(:,2)'), ones(size(angle_matched(:,1)')));

    xyz = [x; y; z];
    xyz = bsxfun(@rdivide, xyz, sqrt(sum(xyz.^2,1)));
    [P, K, voronoiboundary] = voronoisphere(xyz);
    
    plot3(ax, xyz(1,:),xyz(2,:),xyz(3,:),'w.');
    
    n = length(PSD_angle);
    for k = 1:n
        X = voronoiboundary{k};
        %                 cl = [(s1(k)*0.99),(0.2 - (s1(k)*0.2)),(1 - (s1(k)))]; % smallest solid angles in blue, biggest in red
        %         cl = [1 - s1(k),1 - s1(k),1 - s1(k)]; % smallest solid angles in white, biggest in black
        
        
        for i = 1:length(X(:,1))
            for j = 1:length(X(1,:))
                if isnan(X(i,j))
                    X(i,j) = 0;
%                     ere = 1
                end
            end
        end
        
        
        fill3(X(1,:),X(2,:),X(3,:),PSD_angle(k),'Parent',ax,'EdgeColor','w'); % for white outline / edge colour
%         fill3(X(1,:),X(2,:),X(3,:),s(k),'Parent',ax,'EdgeColor','none');     % for no edge colour / outline
    end
    

    c2 = colorbar;
    c2.Label.String = 'PSD';% (x 10^{-2})';
    
    view([45 18])

    
    xlabel ( 'X axis' )
    ylabel ( 'Y axis' )
    zlabel ( 'Z axis' )
    
    grid on
    grid minor
    ax.MinorGridAlphaMode = 'manual';
    ax.MinorGridAlpha = 0.1;
    ax.FontSize = 14;
    
    colormap(flipud(parula));
    
    axis(ax,'equal');
    axis(ax,[-1 1 -1 1 -1 1]);
    
    
    xticks([-1 0 1])
    yticks([-1 0 1])
    zticks([-1 0 1])

    set(gcf, 'Color', 'w');
    
    dcm_obj = datacursormode;
    set(dcm_obj,'UpdateFcn',@disp_ploar)
    % display spherical coordinates (in degrees) instead of cartesian coordinates
    
    title(['Bernschutz (reconstructed), mean PSD = ' num2str(PSD_mean)])
    
    caxis([0 top]);
    
    rotate3d on
end

if plotflag == 2 || plotflag == 3
    figure
    heatmap_plot(angle_matched(:,1)',angle_matched(:,2)',PSD_angle)
    title(['\fontsize{21}Bernschutz (reconstructed), mean PSD = ' num2str(PSD_mean)])
    xlabel('Azimuth (°)','FontSize',18)
    ylabel('Elevation (°)','FontSize',18)
    colormap(flipud(parula));
    c2 = colorbar;
    c2.Label.String = 'PSD';% (x 10^{-2})';
    caxis([0 top]);
    grid on
    grid minor
    set(gcf, 'Position',  [100, 100, 900, 500])

end
end

