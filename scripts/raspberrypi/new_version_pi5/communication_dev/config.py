
# Acesse o tutorial completo para saber melhor o que cada configuração faz
# https://docs.google.com/document/d/15PnGwGTq5nmBDbfy9q7DuQrY7UBS7NpwSDrwWN_OYr8/edit

# Configurações de câmera
camera_id = 0           # Defina o id da câmera
camera_exposure = -14   # Defina exposição da câmera

# Multiplicadores de deslocamento
deltax_multiplier = 1 # Defina o multiplicador de deslocamento X
deltay_multiplier = 1 # Defina o multiplicador de deslocamento Y

# Configuração de comunicação Serial
usb_com_port = None  # Configure a porta de comunicação serial (Padrão: None, Valores Possíveis: String, Exemplo: "COM4")

# Configurações de estimativa
border_windowing_method = "blackman_harris"  # Aplica o escurecimento nas bordas das imagens
phase_windowing = None                       # Aplica o janelamento no sinal final da fase
