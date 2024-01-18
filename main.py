from project.barysh.barysh import get_barysh, get_barysh_predict
from project.tabysh.tabysh import get_tabysh, get_tabysh_predict
from project.jatysh.jatysh import get_jatysh, get_jatysh_predict
from project.chygysh.chygysh import get_chygysh, get_chygysh_predict
from project.koptuk.koptuk import get_koptuk, get_koptuk_predict
from project.ilik.ilik import get_ilik, get_ilik_predict

barysh_model, barysh_padded_sequences = get_barysh()
tabysh_model, tabysh_padded_sequences = get_tabysh()
jatysh_model, jatysh_padded_sequences = get_jatysh()
chygysh_model, chygysh_padded_sequences = get_chygysh()
koptuk_model, koptuk_padded_sequences = get_koptuk()
ilik_model, ilik_padded_sequences = get_ilik()


while True:
        input_text = list(map(str, input("Атооч  : ").split()))
        get_ilik_predict(input_text, ilik_padded_sequences, ilik_model)
        get_barysh_predict(input_text, barysh_padded_sequences, barysh_model)
        get_tabysh_predict(input_text, tabysh_padded_sequences, tabysh_model)
        get_jatysh_predict(input_text, jatysh_padded_sequences, jatysh_model)
        get_chygysh_predict(input_text, chygysh_padded_sequences, chygysh_model)
        get_koptuk_predict(input_text, koptuk_padded_sequences, koptuk_model)

