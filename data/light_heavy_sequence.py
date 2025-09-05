import re
from typing import Tuple, Optional

class Heavy_Light:

    def _split_by_motif_end(self, seq: str,
                            exact_motifs,
                            regex_motifs=None,
                            fallback_len: Optional[int] = None) -> Tuple[str, str, Optional[int]]:
        """
        Split seq at the *end* of a VL/VH FR4 terminal motif.
        Returns (N_term, C_term, cut_idx), where cut_idx is the index AFTER the motif.
        """
        s = seq.strip().upper()

        # 1) exact motifs (ensure these end exactly at VL terminus)
        for m in exact_motifs:
            idx = s.find(m)
            if idx != -1:
                cut = idx + len(m)  # cut AFTER the motif (e.g., after ...VEIK)
                return seq[:cut], seq[cut:], cut

        # 2) regex motifs
        if regex_motifs:
            for pat in regex_motifs:
                m = re.search(pat, s)
                if m:
                    cut = m.end()
                    return seq[:cut], seq[cut:], cut

        # 3) optional fallback (disabled by default for safety)
        if fallback_len is not None and len(s) >= fallback_len:
            cut = min(fallback_len, len(s))
            return seq[:cut], seq[cut:], cut

        # 4) not found
        print("Motif not found – unable to split.")
        return seq, "", None


    def split_light_chain_constant_region(self, seq: str) -> Tuple[str, str, Optional[int]]:
        """
        Split κ/λ light chain into VL | CL, ensuring VL ends at '...VEIK' (or LEIK/LTVL variants).
        CL begins immediately after, e.g., 'R' in 'RTV...'.
        """
        exact_motifs = [
            # κ common FR4 termini (end of VL):
            "FGGGTKVEIK", "GQGTKVEIK", "FGQGTKVEIK",
            "FGGGTKLEIK", "GQGTKLEIK",
            "FGGGTKLTVL", "GQGTKLTVL",
        ]
        # Regex variants: end strictly with VEIK/LEIK/LTVL
        regex_motifs = [
            r"(?:F?G[GA]GTK)(?:V|L)EIK",   # ...GGTKVEIK / ...GGTKLEIK (FGG/ GQG variants allowed)
            r"(?:F?G[GA]GTK)LTVL",         # ...GGTKLTVL
        ]
        return self._split_by_motif_end(seq, exact_motifs, regex_motifs, fallback_len=None)


    def split_heavy_chain_constant_region(self, seq: str) -> Tuple[str, str, Optional[int]]:
        """
        Split heavy chain into VH | CH1+ at common FR4 end motifs.
        (Unchanged logic; keeps VH ending at WGQGT...TVS[STA] variants.)
        """
        exact_motifs = [
            "WGQGTLVTVSS", "WGQGTTVTVSS", "WGQGTLVTVSA",
            "WGQGTLVTVST", "WGGGTLVTVSS", "WGAGTTVTVSS", "WGQGTLITVSS",
        ]
        regex_motifs = [
            r"W(GQ|GA|GG)GT.{1,2}TVS[STA]"
        ]
        return self._split_by_motif_end(seq, exact_motifs, regex_motifs, fallback_len=110)



HeavyLight = Heavy_Light()
# Light chain example
light_seq = "DIQLTQSPSSLSASVGDRVTITCSASQDISNYLNWYQQKPGKAPKVLIYFTSSLHSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYSTVPWTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
heavy_seq = "EVQLVESGGGLVQPGGSLRLSCAASGYDFTHYGMNWVRQAPGKGLEWVGWINTYTGEPTYAADFKRRFTFSLDTSKSTAYLQMNSLRAEDTAVYYCAKYPYYYGTSHWYFDVWGQGTLVTVSSASTKGPSVFPLAPSSKSTSGGTAALGCLVKDYFPEPVTVSWNSGALTSGVHTFPAVLQSSGLYSLSSVVTVPSSSLGTQTYICNVNHKPSNTKVDKKVEPKSCDKTHL"
vh, ch1, cut_H = HeavyLight.split_heavy_chain_constant_region(heavy_seq)
vl, cl, cut_L  = HeavyLight.split_light_chain_constant_region(light_seq)

print("VH Heavy:\n", vh)
print("VL Light:\n", vl)

print("cut light idx:", cut_L)
print("cut heacy idx:", cut_H)
# Heavy chain example

print("CL:\n", cl)
print("CH1+:\n", ch1)

