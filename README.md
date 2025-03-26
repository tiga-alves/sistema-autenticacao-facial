# Sistema de Autenticação Facial com Detecção de Vivacidade

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68%2B-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red)
![Face Recognition](https://img.shields.io/badge/Face--Recognition-1.3.0-orange)

## 📋 Sumário
- [Sobre o Projeto](#-sobre-o-projeto)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Arquitetura](#-arquitetura)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Segurança](#-segurança)
- [Contribuição](#-contribuição)
- [Licença](#-licença)

## 🎯 Sobre o Projeto

Este sistema de autenticação facial foi desenvolvido para resolver um problema crítico de segurança em serviços bancários, onde foram identificados casos de fraude mesmo com a autenticação tradicional por senha. O sistema implementa uma camada adicional de segurança através do reconhecimento facial com detecção de vivacidade, garantindo que apenas usuários autênticos possam acessar serviços sensíveis.

### Contexto do Problema
- Clientes reportando serviços não contratados
- Autenticações por senha sendo comprometidas
- Necessidade de reembolsos e contenção de processos judiciais

## ✨ Funcionalidades

### 1. Detecção Facial
- Identificação precisa de faces em tempo real
- Processamento otimizado de imagens
- Suporte para múltiplos ângulos e condições de iluminação

### 2. Reconhecimento Facial
- Comparação com banco de dados de usuários cadastrados
- Alta precisão na identificação
- Tolerância configurável para matches

### 3. Detecção de Vivacidade (Liveness Detection)
- Prevenção contra ataques com fotos estáticas
- Análise de qualidade de imagem
- Verificação de movimento natural

## 🛠 Tecnologias Utilizadas

- **Python 3.8+**: Linguagem base do projeto
- **FastAPI**: Framework web de alta performance
- **OpenCV**: Processamento de imagens e visão computacional
- **face-recognition**: Biblioteca para reconhecimento facial
- **MediaPipe**: Detecção facial em tempo real
- **NumPy**: Processamento numérico e arrays
- **Streamlit**: Interface gráfica interativa

## 🏗 Arquitetura

O projeto segue uma arquitetura modular com os seguintes componentes:

```
face_authentication/
├── app/
│   └── app.py           # Interface Streamlit
├── database/            # Banco de faces cadastradas
├── custom_face_detection.py
├── face_identification.py
├── liveness_detection.py
├── main.py             # API FastAPI
└── requirements.txt
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/face_authentication.git
cd face_authentication
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate   # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## 💻 Como Usar

1. Prepare o banco de faces:
   - Adicione as imagens dos usuários na pasta `database/`
   - Nomeie as imagens com identificadores únicos

2. Inicie o servidor FastAPI:
```bash
uvicorn main:app --reload
```

3. Para a interface gráfica, execute:
```bash
streamlit run app/app.py
```

## 🔒 Segurança

- Implementação de detecção de vivacidade para prevenir ataques com fotos
- Tolerância configurável para matches faciais
- Logs de tentativas de autenticação
- Tratamento seguro de exceções

## 🤝 Contribuição

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 📊 Métricas e Performance

- Tempo médio de processamento: < 500ms
- Suporte para múltiplas faces simultâneas

## 🔍 Troubleshooting

### Problemas Comuns

1. **Erro na detecção facial**
   - Verifique a iluminação
   - Garanta que o rosto está bem enquadrado
   - Verifique a qualidade da câmera

2. **Falha na autenticação**
   - Confirme se a face está cadastrada
   - Verifique o valor de FACE_MATCH_TOLERANCE
   - Tente atualizar a foto do banco de dados

## 📞 Suporte

Para suporte e dúvidas, por favor abra uma issue no GitHub ou entre em contato através do [seu-email@dominio.com].

---
Desenvolvido com ❤️ para segurança bancária 