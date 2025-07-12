# Modifiche fatte rispetto al paper:



1. Nel paper il è CNNblock costituito d LeakyReLU -> BN -> CNN. Ora invece è \*\*CNN -> BN -> SiLU\*\*
2. Inserito self attention tra i ResBlock nel Feature extractor. La sequenza: ResBlock -> Att combina il potere gerarchico locale delle convoluzioni con la visione globale dell’attenzione. La SelfAttention consente a ogni punto (h, w) di "guardare" tutti gli altri punti del feature map. Questo è utile se l’informazione è sparsa nello spazio (es: oggetti non contigui) e quanto L’importanza relativa tra regioni non è locale (es: correlazione tra bordi distanti). L’attenzione pesa dinamicamente le feature, rendendo il modello Più adattive e Più capace di dare priorità a zone rilevanti in base al contesto
3. Modifica di (CNNBlock3 + CNNBlock4): ora l'output finale è B x 512 x 3 x 6  
