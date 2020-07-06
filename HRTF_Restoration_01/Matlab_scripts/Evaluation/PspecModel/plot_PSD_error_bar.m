function  plot_PSD_error_bar(psdResultsAMB,psdResultsAIO, ciAmbPos,ciAmbNeg,ciAIOPos,ciAIONeg)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

dist = 0.08;

psdResultsAMBX = (1:length(psdResultsAMB))-dist;
psdResultsAIOX = (1:length(psdResultsAMB))+dist;

combined = [psdResultsAMB psdResultsAIO];
combinedX = [psdResultsAMBX psdResultsAIOX];

[combinedX, sortOrder] = sort(combinedX,'ascend');
combined = combined(sortOrder);

    plotColour = get(gca,'colororder');



h  = figure('Color', [1 1 1]); 
p1 = plot(psdResultsAMBX,psdResultsAMB,'s','Color',plotColour(1,:));
set(p1, 'MarkerSize', 10,'LineWidth',1.6,'MarkerFaceColor',plotColour(1,:))

hold on
p2 = plot(psdResultsAIOX,psdResultsAIO,'s','Color',plotColour(2,:));
set(p2, 'MarkerSize', 10,'LineWidth',1.6,'MarkerFaceColor',plotColour(2,:))


for i = 1:(length(psdResultsAMB)-1)
line([i+0.5 i+0.5],[0 3],'Color',[0.8 0.8 0.8],'LineStyle',':','LineWidth',0.2);
end

% p2 = plot(psdResultsAIOX,psdResultsAIO,'b^');
% set(p2, 'MarkerSize', 9,'LineWidth',1.3)

set(gca,'fontsize', 18);
% e1 = plot(combinedX,combined,'k'); % to put a line in between each result



for i = 1:(length(psdResultsAMB)-1)
line([i+0.5 i+0.5],[-5 105],'Color',[0.8 0.8 0.8],'LineStyle',':','LineWidth',0.2);
end



er1 = errorbar(psdResultsAMBX,psdResultsAMB,ciAmbNeg,ciAmbPos,'.'); 
er2 = errorbar(psdResultsAIOX,psdResultsAIO,ciAIONeg,ciAIOPos,'.'); 
er1.Color = [0.3 0.3 0.3];
er2.Color = [0.3 0.3 0.3];
er1.LineWidth = 1.1;
er2.LineWidth = 1.1;

p1 = plot(psdResultsAMBX,psdResultsAMB,'s','Color',plotColour(1,:));
set(p1, 'MarkerSize', 10,'LineWidth',1.6,'MarkerFaceColor',plotColour(1,:))
p2 = plot(psdResultsAIOX,psdResultsAIO,'s','Color',plotColour(2,:));
set(p2, 'MarkerSize', 10,'LineWidth',1.6,'MarkerFaceColor',plotColour(2,:))



ylabel('PSD Score');
xlabel('HRTF dataset');
set(gca,'YGrid','on') 
set(gca,'YMinorGrid','on') 
xticks([1 2 3 4 5 6 7 8 9 10]);
set(gca,'xticklabels',{'SADIE 18 (training data)','SADIE 19 (test data)','SADIE 20 (test data)', 'Bernschutz (test data)'})

legend({'SH input', 'Model output'}, 'location', 'northeast');

xlim([0.5 (length(psdResultsAMB)+0.5)])
ylim([0 6.5])
% pbaspect([1 1 1])

%  pbaspect([0.75 1 1])


% set(gca, 'YDir', 'reverse')


hold off

end

