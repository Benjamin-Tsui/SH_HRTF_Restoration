function SOFAplotGeometry(Obj, index, plot_source_label)
% SOFAplotGeometry(Obj) plots the geometry found in the Obj.
%
% SOFAplotGeometry(Obj, index) plots the geometry for the measurements
% given in the index.

% Copyright (C) 2012-2013 Acoustics Research Institute - Austrian Academy of Sciences;
% Licensed under the EUPL, Version 1.1 or ? as soon they will be approved by the European Commission - subsequent versions of the EUPL (the "License")
% You may not use this work except in compliance with the License.
% You may obtain a copy of the License at: http://joinup.ec.europa.eu/software/page/eupl
% Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" basis, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing  permissions and limitations under the License.

global legendEntries
global legendStrings

if nargin < 3
    plot_source_label = 0;
end

switch Obj.GLOBAL_SOFAConventions
    %%
    case {'SimpleFreeFieldHRIR','SingleRoomDRIR','SimpleFreeFieldTF'}
        
        legendStrings = {'ListenerPosition','ListenerView','Receivers','SourcePosition'};
        
        if ~exist('index','var') || isempty(index)
            index = nonzeros(Obj.ListenerSourcePairIndex);
        end
        
        % Expand entries to the same number of measurement points
        Obj = SOFAexpand(Obj);
        
        % See if the room geometry is specified
        if strcmp(Obj.GLOBAL_RoomType,'shoebox')
            figure('Position',[1 1 (Obj.RoomCornerB(1)-Obj.RoomCornerA(1))*1.2 Obj.RoomCornerB(2)-Obj.RoomCornerA(2)]*100);
            box on; hold on; h=[];
            % plot the room
            rectangle('Position',[Obj.RoomCornerA(1) ...
                Obj.RoomCornerA(2) ...
                Obj.RoomCornerB(1)-Obj.RoomCornerA(1) ...
                Obj.RoomCornerB(2)-Obj.RoomCornerA(2)]);hold on;
            rectangle('Position',[Obj.RoomCornerA(1) ...
                Obj.RoomCornerA(2) ...
                Obj.RoomCornerB(1)-Obj.RoomCornerA(1) ...
                Obj.RoomCornerB(2)-Obj.RoomCornerA(2)]);hold on;
            
        else
            figure; hold on;
        end
        legendEntries = [];
        title(sprintf('%s, %s',Obj.GLOBAL_SOFAConventions,Obj.GLOBAL_RoomType));
        % Get ListenerPosition, ListenerView, ReceiverPosition, and SourcePosition
        % NOTE: ListenerPosition is set to [0 0 0] for SimpleFreeFieldHRIR
        LP = SOFAconvertCoordinates(Obj.ListenerPosition(index,:),Obj.ListenerPosition_Type,'cartesian');
        LV = SOFAconvertCoordinates(Obj.ListenerView(index,:),Obj.ListenerView_Type,'cartesian');
        RP = SOFAconvertCoordinates(Obj.ReceiverPosition(:,:,index),Obj.ReceiverPosition_Type,'cartesian');
        S  = SOFAconvertCoordinates(Obj.SourcePosition(index,:),Obj.SourcePosition_Type,'cartesian');
        % Use only unique listener and source positions
        uniquePoints = unique([LP LV S],'rows','stable');
        LP = uniquePoints(:,1:3);
        LV = uniquePoints(:,4:6);
        S  = uniquePoints(:,7:9);
        % Plot ListenerPosition
        legendEntries(end+1) = plot3(LP(:,1),LP(:,2),LP(:,3),'ro','MarkerFaceColor',[1 0 0]);
        % Plot ListenerView
        for ii=1:size(LV,1)
            % Scale size of ListenerView vector smaller
            LV(ii,:) = 0.2*LV(ii,:)./norm(LV(ii,:));
            % Plot line for ListenerView vector
            line([LP(ii,1), LV(ii,1)+LP(ii,1)], [LP(ii,2) LV(ii,2)+LP(ii,2)], 'Color',[1 0 0]);
        end
        legendEntries(end+1) = plot3(LV(:,1),LV(:,2),LV(:,3),'ro','MarkerFaceColor',[1 1 1]);
        % Plot ReceiverPositon (this is plotted only for the first ListenerPosition)
        if ndims(RP)>2
            % If ReceiverPositon has more than two dimesnions reduce it to the first
            % ListenerPosition
            RP = shiftdim(RP,2);
            RP = squeeze(RP(1,:,:));
        end
        legendEntries(end+1) = plot3(LP(1,1)+RP(1,1), LP(1,2)+RP(1,2), LP(1,3)+RP(1,3),'rx');
        for ii=2:size(RP,1)
            plot3(LP(1,1)+RP(ii,1), LP(1,2)+RP(ii,2), LP(1,3)+RP(ii,3),'rx');
        end
        % Plot SourcePosition
        %if ~isfield(Obj,'GLOBAL_OriginalSOFAlabels')
        legendEntries(end+1)=plot3(S(:,1),S(:,2),S(:,3),'k.');
        if plot_source_label
            for k = 1:length(S)
                text(S(k,1),S(k,2),S(k,3)+0.1,sprintf('%i',k));
            end
        end
        %     else
        %         labels = Obj.GLOBAL_OriginalSOFAlabels;
        %         legendStrings = legendStrings(1:end-1);
        %         markers = {'k.','m.','g.','r.','c.','y.','k+','ko','k>','k<','kp','kh'};
        %         list = flipud(unique(labels(index,:),'rows','stable'));
        %         labels_cell = cellstr(labels);
        %         for k = 1:size(list,1)
        %             source_position_index = all(ismember(labels_cell,strtrim(list(k,:))),2);lidx = find(source_position_index);
        %             legendEntries(end+1)=plot3(S(lidx,1),S(lidx,2),S(lidx,3),markers{k});
        %             legendStrings = [legendStrings, ['SourcePosition ' strtrim(list(k,:))]];
        %         end
        %     end
        
        legend(legendEntries,legendStrings,'Location','NorthEastOutside');
        xlabel(['X / ' Obj.ListenerPosition_Units]);
        ylabel(['Y / ' Obj.ListenerPosition_Units]);
        zlabel(['Z / ' Obj.ListenerPosition_Units]);
        % Set fixed aspect ratio
        axis equal;
        
    case {'AmbisonicsDRIR'}
        
        switch Obj.GLOBAL_SOFAConventionsVersion
            
            case '1.0'
                
                legendStrings = {'ListenerPosition','ListenerView','EmitterPosition','EmitterView'};
                
                if ~exist('index','var')
                    index=1:Obj.API.E;
                end
                
                
                % Expand entries to the same number of measurement points
                %Obj = SOFAexpand(Obj);
                % See if the room geometry is specified
                if strcmp(Obj.GLOBAL_RoomType,'shoebox')
                    figure('Position',[1 1 (Obj.RoomCornerB(1)-Obj.RoomCornerA(1))*1.2 Obj.RoomCornerB(2)-Obj.RoomCornerA(2)]*100);
                    box on; hold on; h=[];
                    % plot the room
                    rectangle('Position',[Obj.RoomCornerA(1) ...
                        Obj.RoomCornerA(2) ...
                        Obj.RoomCornerB(1)-Obj.RoomCornerA(1) ...
                        Obj.RoomCornerB(2)-Obj.RoomCornerA(2)]);
                else
                    figure; hold on;
                end
                legendEntries = [];
                title(sprintf('%s, %s, %s',Obj.GLOBAL_SOFAConventions,Obj.GLOBAL_RoomType,Obj.GLOBAL_DatabaseName));
                
                LP = SOFAconvertCoordinates(Obj.ListenerPosition,Obj.ListenerPosition_Type,'cartesian');
                LV = SOFAconvertCoordinates(Obj.ListenerView,Obj.ListenerView_Type,'cartesian');
                
                %         if size(LP,1) == 1
                %             E = SOFAconvertCoordinates(Obj.EmitterPosition(index,:),Obj.EmitterPosition_Type,'cartesian');
                %         else
                %             if ndims(squeeze(Obj.EmitterPosition)) == 3
                %                 E = SOFAconvertCoordinates(transpose(reshape(permute(Obj.EmitterPosition,[2 1 3]),3,Obj.API.E*Obj.API.M)),Obj.EmitterPosition_Type,'cartesian');
                %             else
                %                 E  = SOFAconvertCoordinates(Obj.EmitterPosition(index,:,:),Obj.EmitterPosition_Type,'cartesian');
                %             end
                %         end
                
                % Plot ListenerPosition
                for jj = 1:size(LP,3)
                    legendEntries(1) = scatter3(LP(:,1,jj),LP(:,2,jj),LP(:,3,jj),'filled','MarkerFaceColor',[0.8500 0.3250 0.0980]);
                end
                
                % Plot ListenerView
                for jj = 1:size(LV,3)
                    for ii=1:size(LV,1)
                        % Scale size of ListenerView vector smaller
                        LV(ii,:,jj) = 0.5*LV(ii,:,jj)./norm(LV(ii,:,jj));
                        % Plot line for ListenerView vector
                        line([LP(ii,1,jj), LV(ii,1,jj)+LP(ii,1,jj)], [LP(ii,2,jj) LV(ii,2,jj)+LP(ii,2,jj)], [LP(ii,3,jj), LV(ii,3,jj)+LP(ii,3,jj)], 'Color',[0.8500 0.3250 0.0980]);
                    end
                    legendEntries(2) = scatter3(LV(:,1,jj)+LP(:,1,jj),LV(:,2,jj)+LP(:,2,jj),LV(:,3,jj)+LP(:,3,jj),'MarkerEdgeColor',[0.8500 0.3250 0.0980],'MarkerFaceAlpha',0);
                end
                
                SP = SOFAconvertCoordinates(Obj.SourcePosition,Obj.SourcePosition_Type,'cartesian');
                EP = SOFAconvertCoordinates(Obj.EmitterPosition,Obj.EmitterPosition_Type,'cartesian');
                
                % Plot EmitterPosition
                for jj = 1:size(EP,3)
                    legendEntries(3) = scatter3(EP(:,1,jj)+SP(:,1),EP(:,2,jj)+SP(:,2),EP(:,3,jj)+SP(:,3),'filled','MarkerFaceColor',[0 0.4470 0.7410]);
                end
                
                EV = SOFAconvertCoordinates(Obj.EmitterView,Obj.EmitterView_Type,'cartesian');
                
                %Plot EmitterView
                for jj = 1:size(EV,3)
                    for ii=1:size(EV,1)
                        % Scale size of EmitterView vector smaller
                        EV(ii,:,jj) = 0.5*EV(ii,:,jj)./norm(EV(ii,:,jj));
                        % Plot line for EmitterView vector
                        line([EP(ii,1,jj)+SP(:,1), EV(ii,1,jj)+EP(ii,1,jj)+SP(:,1)], [EP(ii,2,jj)+SP(:,2) EV(ii,2,jj)+EP(ii,2,jj)+SP(:,2)], [EP(ii,3,jj)+SP(:,3), EV(ii,3,jj)+EP(ii,3,jj)+SP(:,3)],'LineStyle','--');
                    end
                    legendEntries(4) = scatter3(EV(:,1,jj)+EP(:,1,jj)+SP(:,1),EV(:,2,jj)+EP(:,2,jj)+SP(:,2),EV(:,3,jj)+EP(:,3,jj)+SP(:,3),'MarkerEdgeColor',[0 0.4470 0.7410],'MarkerFaceAlpha',0);
                end
                
                legend(legendEntries,legendStrings,'Location','NorthEastOutside');
                xlabel(['X / ' Obj.ListenerPosition_Units]);
                ylabel(['Y / ' Obj.ListenerPosition_Units]);
                zlabel(['Z / ' Obj.ListenerPosition_Units]);
                
                view(-30,25);
                % Set fixed aspect ratio
                axis equal;
                
                if strcmp(Obj.GLOBAL_RoomType,'shoebox')
                    xlim([0 Obj.RoomCornerB(1)]);
                    ylim([0 Obj.RoomCornerB(2)]);
                    zlim([0 Obj.RoomCornerB(3)]);
                end
                
            case '2.0'
                
                % Idea how to restrict the plot to some particular positions: plot
                % the source positions that correspond to the values in 'index',
                % and look for the corresponding 'row' of ListenerSourcePairIndex containing this
                % value in order to plot the associated mic position
                
                ListenerSourcePairIndex = Obj.ListenerSourcePairIndex;
                M = size(ListenerSourcePairIndex,1);
                nof_mic_pos = length(nonzeros(any(ListenerSourcePairIndex,2)));
                
                if ~exist('index','var')
                    index = (1:nof_mic_pos);
                end
                
                % Expand entries to the same number of measurement points
                % See if the room geometry is specified
                if strcmp(Obj.GLOBAL_RoomType,'shoebox') && (any(Obj.RoomCornerA) || any(Obj.RoomCornerB))
                    figure('Position',[1 1 (Obj.RoomCornerB(1)-Obj.RoomCornerA(1))*1.2 Obj.RoomCornerB(2)-Obj.RoomCornerA(2)]*100);
                    box on; hold on; h=[];
                    title(sprintf('%s',Obj.GLOBAL_DatabaseName));
                    % plot the room
                    A1 = Obj.RoomCornerA;
                    B1 = Obj.RoomCornerB;
                    A2 = [A1(1) B1(2) A1(3)];
                    B2 = [B1(1) A1(2) B1(3)];
                    A3 = [A1(1) B1(2) B1(3)];
                    B3 = [B1(1) A1(2) A1(3)];
                    A4 = [A1(1) A1(2) B1(3)];
                    B4 = [B1(1) B1(2) A1(3)];
                    patch('Faces',[1 2 3 4;5 6 7 8;1 2 8 7;4 3 5 6;1 4 6 7;2 3 5 8],'Vertices',[A1;A2;A3;A4;B1;B2;B3;B4]);alpha(0.05);
                else
                    figure; hold on;
                end
                view(-45,30);
                title(sprintf('%s',Obj.GLOBAL_DatabaseName));
                
                LP = SOFAconvertCoordinates(Obj.ListenerPosition,Obj.ListenerPosition_Type,'cartesian');
                LV = SOFAconvertCoordinates(Obj.ListenerView,Obj.ListenerView_Type,'cartesian');
                if size(LV,1) == 1
                    LV = repmat(LV,M,1);
                end
                
                SP = SOFAconvertCoordinates(Obj.SourcePosition,Obj.SourcePosition_Type,'cartesian');
                SV = SOFAconvertCoordinates(Obj.SourceView,Obj.SourceView_Type,'cartesian');
                if size(SV,1) == 1
                    SV = repmat(SV,M,1);
                end
                
                % Plot Source and Listener Positions, plus Source and Listener View
                
                markers = {'+','x','v','s','d','*','^','>','<'};
                legendEntries = [];
                legendStrings = {};
                legend_idx = 1;
                
                for mic_pos_idx = index                    
                    
                    LP_idx = ListenerSourcePairIndex(mic_pos_idx,~isnan(ListenerSourcePairIndex(mic_pos_idx,:)));
                    legendEntries(2*(legend_idx-1)+1) = scatter3(LP(LP_idx(1),1),LP(LP_idx(1),2),LP(LP_idx(1),3),80,char(markers(mic_pos_idx)),'MarkerEdgeColor','r');
                    line([LP(LP_idx(1),1), LP(LP_idx(1),1)+LV(LP_idx(1),1)], [LP(LP_idx(1),2), LP(LP_idx(1),2)+LV(LP_idx(1),2)],[LP(LP_idx(1),3), LP(LP_idx(1),3)+LV(LP_idx(1),3)], 'Color','r','LineStyle','--');
                    scatter3(LP(LP_idx(1),1)+LV(LP_idx(1),1),LP(LP_idx(1),2)+LV(LP_idx(1),2),LP(LP_idx(1),3)+LV(LP_idx(1),3),20,'MarkerEdgeColor','r','MarkerFaceAlpha',0);
                    legendStrings{2*(legend_idx-1)+1} = sprintf('Listener position %i',mic_pos_idx);
                    
                    for lpk_pos_idx = LP_idx
                        legendEntries(2*legend_idx) = scatter3(SP(lpk_pos_idx,1),SP(lpk_pos_idx,2),SP(lpk_pos_idx,3),80,char(markers(mic_pos_idx)),'MarkerEdgeColor','k');
                        line([SP(lpk_pos_idx,1), SP(lpk_pos_idx,1)+SV(lpk_pos_idx,1)], [SP(lpk_pos_idx,2), SP(lpk_pos_idx,2)+SV(lpk_pos_idx,2)],[SP(lpk_pos_idx,3), SP(lpk_pos_idx,3)+SV(lpk_pos_idx,3)], 'Color','k','LineStyle','--');
                        scatter3(SP(lpk_pos_idx,1)+SV(lpk_pos_idx,1),SP(lpk_pos_idx,2)+SV(lpk_pos_idx,2),SP(lpk_pos_idx,3)+SV(lpk_pos_idx,3),20,'MarkerEdgeColor','k','MarkerFaceAlpha',0);
                        legendStrings{2*legend_idx} = sprintf('Source positions associated to Listener position %i',mic_pos_idx);
                    end
                    
                    legend_idx = legend_idx + 1;
                    
                end
                axis equal;
                xlim([A1(1)-0.5 B1(1)+0.5]);
                ylim([A1(2)-0.5 B1(2)+0.5]);
                zlim([A1(3)-0.5 B1(3)+0.5]);
                legend(legendEntries,legendStrings);
                
        end
        
    otherwise
        error('This SOFAConventions is not supported for plotting');
end


% Add a little bit extra space at the axis
if ~strcmp(Obj.GLOBAL_RoomType,'shoebox')
    
    axisLimits = axis();
    paddingSpace = 0.2 * max(abs(axisLimits(:)));
    axisLimits([1 3]) = axisLimits([1 3]) - paddingSpace;
    axisLimits([2 4]) = axisLimits([2 4]) + paddingSpace;
    axis(axisLimits);
    
end
grid on;