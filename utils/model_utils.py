def create_model(max_len, labels, learning_rate=5e-5, experiment='5f', subexp='freezeBoth'):
    encoder = TFBertModel.from_pretrained("bert-base-uncased")
    print(experiment)

    if experiment == '5n':
        print('Freezing BERT weights')

        if subexp == 'freezeEmbeddings':
            print('freezing embeddings')
            encoder.bert.embeddings.trainable = False
        elif subexp == 'freezeEncoders':
            print('freezing encoder layers')
            for layer in encoder.bert.encoder.layer:
                layer.trainable = False
        elif subexp == 'freezeBoth':
            print('freezing embeddings')
            encoder.bert.embeddings.trainable = False
            print('freezing encoder layers')
            for layer in encoder.bert.encoder.layer:
                layer.trainable = False
        else:
            assert False, 'Unrecognized sub experiment type'

    input_ids = layers.Input(shape=(max_len,), dtype=tf.int32)
    attention_mask = layers.Input(shape=(max_len,), dtype=tf.int32)

    embedding = encoder(
        input_ids, attention_mask=attention_mask
    )['pooler_output']

    dense = layers.Dense(1024, activation='relu')(embedding)
    out = layers.Dense(len(labels), activation='softmax')(dense)

    model = keras.Model(
        inputs=[input_ids, attention_mask],
        outputs=out, )

    loss = keras.losses.SparseCategoricalCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model
