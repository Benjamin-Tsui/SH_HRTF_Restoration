function s = GetVoronoiPlotandSolidAng(azimuth, elevation, plotflag)

% Returns values of solid angles from arrays of azimuth and elevation
% angles in degrees.
%
% ========================
% Inputs:
%     azimuth:            array of azimuth angles in degrees
%
%     elevation           array of elevation angles in degrees
%
%     plotFlag:           Enter 1 to plot a voronoi sphere plot
%
% ================ Tom McKenzie, University of York, 2018 ================
% Version 1.0 - 5/3/2018

[x, y, z] = sph2cart(deg2rad(azimuth), deg2rad(elevation), ones(size(azimuth)));

xyz = [x; y; z];
xyz = bsxfun(@rdivide, xyz, sqrt(sum(xyz.^2,1)));
n = length(azimuth);
[P, K, voronoiboundary] = voronoisphere(xyz);


% Compute Solid Angles
s = vcell_solidangle(P, K);
% s1 = scaledata(s,0,1);


s = s / sum(s); % so that the sum of all the solid angles = 1.

%% Graphic
if plotflag == 1
    
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
    
    plot3(ax, xyz(1,:),xyz(2,:),xyz(3,:),'w.');
    
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
        
        
        fill3(X(1,:),X(2,:),X(3,:),s(k),'Parent',ax,'EdgeColor','w'); % for white outline / edge colour
%         fill3(X(1,:),X(2,:),X(3,:),s(k),'Parent',ax,'EdgeColor','none');     % for no edge colour / outline
    end
    

    c2 = colorbar;
    c2.Label.String = 'Solid Angle';% (x 10^{-2})';
    
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
    
end

