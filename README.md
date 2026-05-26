# Tradutor de Libras para Português via Visão Computacional (MVP)

> **Aviso:** Este projeto é um protótipo (Produto Mínimo Viável - MVP) desenvolvido como parte dos requisitos da disciplina **IA750 - Engenharia de Reabilitação** da Universidade Estadual de Campinas (UNICAMP). 

Este repositório contém a implementação de um sistema de Visão Computacional capaz de reconhecer sinais estáticos da Língua Brasileira de Sinais (Libras) em tempo real, utilizando uma webcam convencional, e traduzi-los para texto em português.

---

## 1. Contexto e Motivação
A surdez atinge milhões de brasileiros (cerca de 2,7 milhões possuem surdez profunda). Apesar de a legislação garantir o direito à igualdade de oportunidades e não discriminação, a comunidade surda enfrenta barreiras diárias devido à:
* Falta e precarização de intérpretes de Libras no país.
* Dificuldade no uso do português escrito, que atua como uma segunda língua (ausência de grafofonêmica e estruturas gramaticais diferentes da Libras).

Este projeto nasce da necessidade de criar tecnologias assistivas acessíveis (Engenharia de Reabilitação) que promovam a autonomia de pessoas surdas em ambientes como hospitais, escolas e serviços públicos.

---

## 2. O Protótipo (Escopo Atual)
A Libras é uma língua complexa que envolve sinais dinâmicos, oclusão de mãos, trajetória no espaço e expressões faciais. 

Para validar a viabilidade técnica da extração de características geométricas, este **protótipo inicial foca exclusivamente na Classificação Estática** (dactilologia e sinais manuais fixos). 

**Tecnologias Empregadas:**
* **Extração de Características:** Transição do uso ineficiente de "pixels brutos" para o mapeamento geométrico leve através da extração de *Landmarks* espaciais (coordenadas X, Y, Z) utilizando o **MediaPipe (Tasks API)**.
* **Classificação (Machine Learning):** Treinamento de um algoritmo clássico (Random Forest via `scikit-learn`) alimentado apenas pelas matrizes de coordenadas, garantindo inferência em milissegundos mesmo em computadores sem placa de vídeo dedicada.

---

## 3. Estrutura do Repositório

O projeto foi construído seguindo um pipeline modular de processamento de dados:

```text
/
├── data/                  # Armazena o dataset criado (dataset_libras.csv)
├── models/                # Contém o modelo do MediaPipe (.task) e o modelo treinado (.pkl)
├── src/                   # Código-fonte do pipeline
│   ├── captura_base.py    # Validação da extração de landmarks via webcam
│   ├── coleta_dados.py    # Script para gravação de amostras e construção do dataset
│   ├── treinamento.py     # Treinamento do modelo Random Forest
│   └── tradutor.py        # Aplicação final: inferência e tradução em tempo real
├── requirements.txt       # Dependências do projeto
└── README.md              # Documentação
