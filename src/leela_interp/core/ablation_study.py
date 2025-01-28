import torch
import matplotlib.pyplot as plt
import numpy as np
from leela_interp.tools import figure_helpers as fh
import os
from leela_interp.core.checkmate_study import CheckmateStudy

class AblationStudy:
    def __init__(self, folder_name='', device='cpu', double_branch=False, b=False):
        self.folder_name = folder_name
        self.device = device
        self.b = b
        self.load_ablation_data(double_branch)
        fh.set()

    def load_ablation_data(self, double_branch=False):
        ablation = {}
        results_path = os.path.join("results", self.folder_name)
        for file in os.listdir(results_path):
            suffix = "_ablation_b.pt" if self.b else "_ablation.pt"
            if file.endswith(suffix):
                file_prefix = file[:-len(suffix)]
                if file_prefix == 'single_weight':
                    continue
                if file_prefix != 'other':
                    file_prefix = AblationStudy.pretty_prefix(file_prefix, double_branch)
                ablation[file_prefix] = torch.load(os.path.join(results_path, file), map_location=self.device)
        self.ablation = ablation

    @staticmethod
    def pretty_prefix(prefix, double_branch=False):
        if double_branch:
            parts = prefix.split('_to_')
            if len(parts) != 2:
                return prefix
            
            source, target = parts
            source_name, source_num = source.split('_')
            target_name, target_num = target.split('_')
            
            source_number = AblationStudy.word_to_number(source_name)
            target_number = AblationStudy.word_to_number(target_name)
            
            # Convert numbers to letters (1->A, 2->B)
            source_letter = chr(64 + int(source_num))  # 65 is ASCII for 'A'
            target_letter = chr(64 + int(target_num))
            
            return f"{source_number}{source_letter}$\\rightarrow${target_number}{target_letter} target"
        else:
            first_number, _, second_number = prefix.split("_")
            first_number = AblationStudy.word_to_number(first_number) + first_number[-2:]
            second_number = AblationStudy.word_to_number(second_number) + second_number[-2:]
            return first_number + r"$\rightarrow$" + second_number + ' target'

    def plot_ablation_effects(self, mask=None, verbose=False, filename=None, puzzle_set=None, LH=None, axs=None, double_branch=False):
        if mask is None:
            mask = slice(None)

        if double_branch:
            colors = {
                "other": fh.COLORS[-1],
                r"3A$\rightarrow$1A target": fh.COLORS[0],
                r"3A$\rightarrow$1B target": fh.COLORS[1],
                r"3B$\rightarrow$1A target": fh.COLORS[2],
                r"3B$\rightarrow$1B target": fh.COLORS[3],
            }
        else:
            colors = {
                "other": fh.COLORS[-1],
                r"3rd$\rightarrow$1st target": fh.COLORS[0],
                r"5th$\rightarrow$1st target": fh.COLORS[1],
                r"5th$\rightarrow$3rd target": fh.COLORS[2],
                r"7th$\rightarrow$1st target": fh.COLORS[3],
                r"7th$\rightarrow$3rd target": fh.COLORS[4],
                r"7th$\rightarrow$5th target": fh.COLORS[5],
            }

        # Sorted dictionary of ablation
        sorted_ablation = dict(sorted(self.ablation.items(), key=lambda x: x[0]))
        sorted_colors = [colors[key] for key in sorted_ablation.keys()]
        sorted_colors_dict = dict(zip(sorted_ablation.keys(), sorted_colors))

        if len(sorted_ablation) > 4:
            scale_factor = 1.5
        else:
            scale_factor = 1.5

        #print(sorted_ablation)

        fh.set()
        fh.plot_percentiles(
            sorted_ablation,
            zoom_start=94,
            zoom_width_ratio=0.7,
            colors=sorted_colors_dict,
            title=(("Set " + puzzle_set + ", " + LH) if puzzle_set is not None and LH is not None else ""),
            figsize=(fh.get_width(0.66) * scale_factor, 2 * scale_factor),
            tick_frequency=25,
            zoom_tick_frequency=2,
            y_lower=-1 if not double_branch else -2,
            y_upper=4,
            verbose=verbose,
            axs=axs,
        )
        if filename is not None and axs is None:
            fh.save('figures/' + filename, plt.gcf())

    @staticmethod
    def plot_ablation_effects_grid(ablation_configs, n_cols=2, filename=None, tag='', double_branch=False, b=False):
        n_rows = (len(ablation_configs) + n_cols - 1) // n_cols
        figsize = (fh.get_width(0.66)*1., 2*1.)
        figsize = (figsize[0]*n_cols, figsize[1]*n_rows)
        fig, axes = plt.subplots(n_rows, 2*n_cols, figsize=figsize, sharex=False, sharey=True, width_ratios=[1, 0.7]*n_cols)
        axes = np.array(axes).flatten()

        for i, (case, puzzle_set) in enumerate(ablation_configs):
            axs = axes[2*i:2*i+2]
            ablation_study = AblationStudy(folder_name=case + ("_" + tag if tag != '' else "") + "_" + puzzle_set, 
                                         double_branch=double_branch, b=b)
            ablation_study.plot_ablation_effects(filename=None if filename is None else filename + "_" + case + ("_" + tag if tag != '' else "")+ "_" + puzzle_set, 
                                               puzzle_set=puzzle_set, LH=case, axs=axs, double_branch=double_branch)
            if i % n_cols != 0:
                axs[0].set_ylabel('')
                axs[1].set_ylabel('')
            if i // n_cols != n_rows - 1:
                axs[0].set_xlabel('')
                axs[1].set_xlabel('')

        # Hide any unused subplots
        for j in range(2*len(ablation_configs), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if filename is not None:
            fh.save('figures/' + filename, fig)

        plt.show()
        plt.close(fig)

    @staticmethod
    def word_to_number(word):
        ordinal_dict = {
            'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
            'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
            'eleventh': 11, 'twelfth': 12, 'thirteenth': 13, 'fourteenth': 14, 'fifteenth': 15,
            'sixteenth': 16, 'seventeenth': 17, 'eighteenth': 18, 'nineteenth': 19, 'twentieth': 20
        }
        
        return str(ordinal_dict.get(word.lower(), None))
    
class AblationCheckmateStudy(AblationStudy):
    def __init__(self, folder_name='', device='cpu', puzzlename='', load_all=False, double_branch=False, b=False):
        self.folder_name = folder_name
        self.device = device
        self.b = b
        self.load_ablation_data(double_branch)
        fh.set()

        self.puzzlename = puzzlename
        self.checkmate_study = CheckmateStudy(puzzlename=puzzlename, load_all=load_all)
        #print(self.checkmate_study.puzzle_sets['n']['112'])

    def load_ablation_data(self, double_branch=False):
        ablation = {}
        results_path = os.path.join("results", self.folder_name)
        for file in os.listdir(results_path):
            if file.endswith("_ablation.pt"):
                file_prefix = file[:-12]
                if file_prefix == 'single_weight':
                    continue
                if file_prefix != 'other':
                    file_prefix = AblationCheckmateStudy.pretty_prefix(file_prefix, double_branch)
                ablation[file_prefix] = torch.load(os.path.join(results_path, file), map_location=self.device)
        self.ablation = ablation

    def plot_ablation_effects(self, verbose=False, filename=None, puzzle_set=None, LH=None, axs=None, mate=False, double_branch=False):

        n_turns = (len(puzzle_set)+1) // 2
        #print(self.checkmate_study.puzzle_sets)
        mask = self.checkmate_study.puzzle_sets['n'][puzzle_set]["Themes"].apply(lambda x: f"mateIn{n_turns}" in x)
        mask = mask.to_numpy()

        if not mate:
            mask = ~mask

        if np.sum(mask) == 0:
            return

        if double_branch:
            colors = {
                "other": fh.COLORS[-1],
                r"3A$\rightarrow$1A target": fh.COLORS[0],
                r"3A$\rightarrow$1B target": fh.COLORS[1],
                r"3B$\rightarrow$1A target": fh.COLORS[2],
                r"3B$\rightarrow$1B target": fh.COLORS[3],
            }
        else:
            colors = {
                "other": fh.COLORS[-1],
                r"3rd$\rightarrow$1st target": fh.COLORS[0],
                r"5th$\rightarrow$1st target": fh.COLORS[1],
                r"5th$\rightarrow$3rd target": fh.COLORS[2],
                r"7th$\rightarrow$1st target": fh.COLORS[3],
                r"7th$\rightarrow$3rd target": fh.COLORS[4],
                r"7th$\rightarrow$5th target": fh.COLORS[5],
            }

        # Sorted dictionary of ablation
        ablation = {}
        for key, value in self.ablation.items():
            ablation[key] = value[mask]
        sorted_ablation = dict(sorted(ablation.items(), key=lambda x: x[0]))
        sorted_colors = [colors[key] for key in sorted_ablation.keys()]
        sorted_colors_dict = dict(zip(sorted_ablation.keys(), sorted_colors))

        if len(sorted_ablation) > 4:
            scale_factor = 1.5
        else:
            scale_factor = 1.5

        #print(sorted_ablation)

        fh.set()
        try:
            fh.plot_percentiles(
                sorted_ablation,
                zoom_start=94,
                zoom_width_ratio=0.7,
                colors=sorted_colors_dict,
                title=(("Set " + ('M' if mate else 'N') + puzzle_set + ", " + LH) if puzzle_set is not None and LH is not None else ""),
                figsize=(fh.get_width(0.66) * scale_factor, 2 * scale_factor),
                tick_frequency=25,
                zoom_tick_frequency=2,
                y_lower=-1,
                y_upper=YOUR_DESIRED_UPPER_LIMIT,
                verbose=verbose,
                axs=axs,
            )
        except ValueError:
            return
        if filename is not None and axs is None:
            fh.save('figures/' + filename, plt.gcf())

    def plot_ablation_effects_grid(self, ablation_configs, n_cols=2, filename=None, tag='', double_branch=False, b=False):
        n_rows = (len(ablation_configs) + n_cols - 1) // (n_cols)
        figsize = (fh.get_width(0.66)*1., 2*1.)
        figsize = (figsize[0]*2*n_cols, figsize[1]*n_rows)
        fig, axes = plt.subplots(n_rows, 4*n_cols, figsize=figsize, sharex=False, sharey=True, width_ratios=[1, 0.7]*2*n_cols)
        axes = np.array(axes).flatten()

        for i, (case, puzzle_set) in enumerate(ablation_configs):
            for j in range(2):
                axs = axes[4*i+2*j:4*i+2*j+2]
                ablation_study = AblationCheckmateStudy(folder_name=case + ("_" + tag if tag != '' else "") + "_" + puzzle_set, 
                                                      puzzlename=self.puzzlename, double_branch=double_branch, b=b)
                ablation_study.plot_ablation_effects(filename=None if filename is None else filename + "_" + case + ("_" + tag if tag != '' else "")+ "_" + puzzle_set, 
                                                   puzzle_set=puzzle_set, LH=case, axs=axs, mate=j==0, double_branch=double_branch)
                if i % n_cols != 0:
                    axs[0].set_ylabel('')
                    axs[1].set_ylabel('')
                if i // n_cols != n_rows - 1:
                    axs[0].set_xlabel('')
                    axs[1].set_xlabel('')

        # Hide any unused subplots
        for j in range(4*len(ablation_configs), len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        if filename is not None:
            fh.save('figures/' + filename, fig)

        plt.show()
        plt.close(fig)