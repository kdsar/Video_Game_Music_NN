import pretty_midi
import numpy as np
import tensorflow as tf
import os
import collections
import re
import pickle

import matplotlib.pyplot as plt

MIDI_FOLDER = "./data/"

record_file_number = 0
RECORD_PATH = "./tfrecord/"
record_file_name = ("train-%.3d.tfrecords" % record_file_number)

BATCH_SIZE = 1
context = 50
max_song_length_used = context * 10
maxed_parsed_songs = 4000


'''
Aggregate Functions
'''


def prepare_dataset():
    encoded_list = parse_midi_directory(2000, 10000)
    tokenizer = tokenize(encoded_list)
    tokenized_list = get_tensor_from_tokenizer_and_corpus(
        tokenizer, encoded_list)
    create_tfrecord_from_tokenized_list(tokenized_list)


def open_dataset(batch_size):
    dataset = read_tfrecord(batch_size)
    return dataset


'''
MIDI functions
'''


def parse_midi_directory(min_song_length=0, max_song_length=999999999, context_size=context, max_parsed_files=maxed_parsed_songs):
    '''
    Parses the given directory to create tfrecord files using that data

    Parameters
    min_song_length (int): Skip a song if the number of events is smaller than this number
    max_song_length (int): Skip a song if the number of events is bigger than this number
    context_size (int): Amount of events in a single part of the the tfrecord created
    max_parsed_files (int): Maximum number of files parsed

    Returns
    list: A list containing lists of maximum size 'context_size' of events
    '''
    encoded_strings = []
    counter = 0
    errors = 0
    for console_directory in os.listdir(MIDI_FOLDER):
        if console_directory.startswith('.'):
            continue
        console_directory = os.path.join(MIDI_FOLDER, console_directory)
        for game_directory in os.listdir(console_directory):
            if game_directory.startswith('.'):
                continue
            game_directory = os.path.join(console_directory, game_directory)
            for midi_file in os.listdir(game_directory):
                if midi_file.startswith('.'):
                    continue
                try:
                    print(f'Parsing {midi_file}')
                    midi = pretty_midi.PrettyMIDI(
                        os.path.join(game_directory, midi_file))
                    events = midi_to_encoded_string(midi).split(' ')
                    events_len = len(events)
                    if events_len > min_song_length and events_len < max_song_length:
                        for i in range(0, min(events_len, max_song_length_used), context_size):
                            encoded_strings.append(events[i:i + context_size])
                        counter += 1
                    else:
                        raise Exception(
                            f'{game_directory} {midi_file} not in correct size: {len(events)}')
                except Exception as e:
                    errors += 1
                    print(f'Error while parsing {midi_file}: {e}')
                if(counter > max_parsed_files):
                    print(f'{errors} errors occured while parsing')
                    print(
                        f'{counter} files parsed successfully and added to the dataset')
                    return encoded_strings
    print(f'{errors} errors occured while parsing')
    print(f'{counter} files parsed successfully and added to the dataset')
    return encoded_strings


def midi_to_encoded_string(midi, quarter_note_subdivision=4, max_wait=128):
    '''
    Encodes a midi file into a string containing all events
    The string is enclosed between "start" and "end"
    Notes played are represented as words of this form "i{instrument}:v{velocity}:{note pitch}"
    Notes that finish playing will have the same form as above but start with "end:"
    Waiting times have the form wait:{wait amount}
    Tempo changes have the form tempo:{tempo}

    Example: start tempo150 i99:v100:64 wait:8 i99:v110:46 wait:4 end:i99:v100:64 [...] finish

    Volumes are rounded to the closest 10 for a smaller dictionnary

    Parameters:
    midi (PrettyMIDI): A PrettyMidi object of a song
    quarter_note_subdivision (int): How a quarter note should be divided for the smallest interval
    max_wait (int): Do not allow songs with a higher wait time than this number

    Returns:
    string: the encoded string
    '''
    if midi.resolution % quarter_note_subdivision != 0:
        raise Exception(
            f"Invalid subdivion of {quarter_note_subdivision} in resolution {midi.resolution}")

    event_history = {}
    tempo_changes = midi.get_tempo_changes()
    beats_per_second = 60 / tempo_changes[1][0]
    smallest_interval = beats_per_second / (quarter_note_subdivision * 4)

    for i in range(len(tempo_changes[0])):
        event_history[int(midi.time_to_tick(
            tempo_changes[0][i]) / smallest_interval)] = [f'tempo:{int(round(tempo_changes[1][i], -1))}']
    for inst in midi.instruments:
        # if inst.is_drum:
        #     continue
        for note in inst.notes:
            start_time = int(round(note.start / smallest_interval))
            end_time = int(round(note.end / smallest_interval))
            note_velocity = round(note.velocity, -1)
            start_content = f'i{inst.program}:v{note_velocity}:{pretty_midi.note_number_to_name(note.pitch)}'
            end_content = f'end:i{inst.program}:v{note_velocity}:{pretty_midi.note_number_to_name(note.pitch)}'

            if start_time not in event_history:
                event_history[start_time] = []
            if end_time not in event_history:
                event_history[end_time] = []
            event_history[start_time].append(start_content)
            event_history[end_time].append(end_content)

    event_history = collections.OrderedDict(sorted(event_history.items()))
    event_history = remove_end_tempos(event_history)

    encoded_string = 'start '
    moments = ''
    previous_moment = -1
    for moment in event_history:
        if previous_moment != -1:
            wait_amount = moment - previous_moment
            if(wait_amount > max_wait):
                raise Exception(
                    f"Wait of {wait_amount} exceeds maximum allowed of {max_wait}")
            encoded_string += f'wait:{wait_amount} '
        moments += f'{moment} '

        for event in event_history[moment]:
            encoded_string += event + ' '
        previous_moment = moment
    encoded_string = encoded_string[:-1] + ' finish'

    return encoded_string


def remove_end_tempos(event_history):
    '''Some midis finish on a long wait followed by a tempo. Remove these kinds of sections.'''
    r_event_history = collections.OrderedDict(
        reversed(sorted(event_history.items())))
    for moment in r_event_history:
        if(len(r_event_history[moment]) == 1 and r_event_history[moment][0][0:5] == 'tempo'):
            event_history.pop(moment, None)
        else:
            break
    return event_history


def encoded_string_to_midi(string, quarter_note_subdivision=4):
    '''Takes an encoded string of the form generated by midi_to_encoded_string and create a midi file from it'''
    start_location = string.find('start ')
    end_location = string.find(' finish')
    string = string[start_location + 6:end_location]
    # print(string)
    tempo_changes = [[], []]
    notes = []
    current_moment = 0
    events = string.split(' ')
    notes_in_instruments = {}

    for event in events:
        if event.find('tempo:') != -1:
            # 6 characters in 'tempo:'
            tempo = event[6:]
            tempo_changes[0].append(current_moment)
            tempo_changes[1].append(int(tempo))

        elif event.find('wait:') != -1:
            # 5 characters in 'wait:'
            wait = event[5:]
            current_moment += int(wait)
        elif event.find('end:') != -1:
            # 4 characters in 'end:'
            note = event[4:]
            index = next(i for i in range(len(notes)) if notes[i].find(
                note) == 0 and notes[i].find(':e') == -1)
            notes[index] += f':e{current_moment}'
        else:
            notes.append(f"{event}:s{current_moment}")

    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_changes[1][0])

    beats_per_second = 60 / tempo_changes[1][0]
    smallest_interval = beats_per_second / (quarter_note_subdivision * 4)

    for i in range(len(notes)):
        notes[i] = decode_note_notation(notes[i])
        if notes[i][0] not in notes_in_instruments:
            notes_in_instruments[notes[i][0]] = []
        notes[i][3] *= smallest_interval
        notes[i][4] *= smallest_interval
        notes_in_instruments[notes[i][0]].append(notes[i][1:])

    for instrument_program in notes_in_instruments:
        instrument = pretty_midi.Instrument(program=instrument_program)
        for note in notes_in_instruments[instrument_program]:
            instrument.notes.append(pretty_midi.Note(
                velocity=note[0], pitch=note[1], start=note[2], end=note[3]))
        midi.instruments.append(instrument)
    midi.write('test.mid')
    return midi


def decode_note_notation(note_string):
    '''Turn a string note into a list of its attributes'''
    note = re.search('i(.*):v(.*):(.*):s(.*):e(.*)', note_string)
    instrument = int(note.group(1))
    velocity = int(note.group(2))
    pitch = pretty_midi.note_name_to_number(note.group(3))
    start = float(note.group(4))
    end = float(note.group(5))
    return [instrument, velocity, pitch, start, end]


'''
TFRecord functions
'''


def create_tfrecord_from_tokenized_list(tokenized_list):
    '''Create a tfrecord from a tokenized list'''
    writer = tf.io.TFRecordWriter(RECORD_PATH + record_file_name)
    for encoded_note_string in tokenized_list:
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'encoded_note_string': _int64_feature(encoded_note_string)
        }))
        writer.write(tf_example.SerializeToString())
        if(os.path.getsize(RECORD_PATH + record_file_name) > 100000000):
            writer = update_record_file()
    pass


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    if type(value) != list and type(value) != np.ndarray:
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def update_record_file():
    '''
    Set the next record file

    Returns:
    tf.io.TFRecordWriter: the writer class using the new file
    '''
    global record_file_number, record_file_name
    record_file_number += 1
    record_file_name = ("train-%.3d.tfrecords" % record_file_number)
    with open(RECORD_PATH + record_file_name, 'w'):
        pass
    return tf.io.TFRecordWriter(RECORD_PATH + record_file_name)


def read_tfrecord(batch_size=BATCH_SIZE):
    '''
    Read the tfrecord path to get the corresponding dataset

    Parameters:
    batch_size (int): Size used for batching

    Return:
    (Dataset): A batched and shuffled dataset    
    '''
    total_dataset = None
    for tfrecord_file in os.listdir(RECORD_PATH):
        if tfrecord_file.startswith('.'):
            continue
        tfrecord_file = os.path.join(RECORD_PATH, tfrecord_file)
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file)
        if(total_dataset is None):
            total_dataset = raw_dataset
        else:
            total_dataset = total_dataset.concatenate(raw_dataset)
    parsed_dataset = total_dataset.map(
        _parse_data_function).shuffle(10000000).batch(batch_size)
    return parsed_dataset


def _parse_data_function(example_proto):
    '''
    Function used for mapping the tfrecord information to usable data
    '''
    data_feature_description = {
        'encoded_note_string': tf.io.FixedLenFeature([context], tf.int64),
        # 'encoded_note_string': tf.io.VarLenFeature(tf.int64),
    }
    # Parse the input tf.Example proto using the dictionary above.
    features = tf.io.parse_single_example(
        example_proto, data_feature_description)

    return features


'''
Token functions
'''


def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    lang_tokenizer.fit_on_texts(lang)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(lang_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return lang_tokenizer


def get_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
        return pickle.load(handle)


def get_tensor_from_tokenizer_and_corpus(lang_tokenizer, lang):
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor


'''
Testing functions
'''


def pickle_midi_file_dict(pickle_name, min_size=0, max_size=999999999):
    '''
    IGNORE Testing function: create a dict and picle it
    '''
    events_per_midi = {}
    counter = 0
    errors = 0

    for console_directory in os.listdir(MIDI_FOLDER):
        if console_directory.startswith('.'):
            continue
        console_directory = os.path.join(MIDI_FOLDER, console_directory)
        for game_directory in os.listdir(console_directory):
            if game_directory.startswith('.'):
                continue
            game_directory = os.path.join(console_directory, game_directory)
            for midi_file in os.listdir(game_directory):
                if midi_file.startswith('.'):
                    continue
                try:
                    midi = pretty_midi.PrettyMIDI(
                        os.path.join(game_directory, midi_file))
                    events = midi_to_encoded_string(midi).split(' ')
                    if len(events) > min_size and len(events) < max_size:
                        events_per_midi[game_directory + ' ' +
                                        midi_file] = events
                    else:
                        print(
                            f'{game_directory} {midi_file} not in correct size: {len(events)}')
                except Exception as e:
                    errors += 1
                    print(f'Error while parsing {midi_file}: {e}')
    print(f'{errors} errors occured while parsing')

    with open(f'{pickle_name}.pickle', 'wb') as handle:
        pickle.dump(events_per_midi, handle, protocol=pickle.HIGHEST_PROTOCOL)


def check_midi_file_lengths(pickle_name):
    '''
    IGNORE Testing function: Get a pickled dict and save a graph of the lengths
    '''
    with open(f'{pickle_name}.pickle', 'rb') as handle:
        events_per_midi = pickle.load(handle)
        sizes = []
        for k in sorted(events_per_midi, key=lambda x: len(events_per_midi[x]), reverse=True):
            print(k)
            print(len(events_per_midi[k]))
            sizes.append(len(events_per_midi[k]))
        print(len(events_per_midi))
        plt.figure()
        plt.plot(sizes)
        plt.ylabel('Number of events')
        plt.savefig(f'{pickle_name}_graph.png')
        plt.close()

    return events_per_midi


def check_event_sparcity(pickle_name):
    '''
    IGNORE Testing function: Get a pickled dict and print all events with the number of times they appear
    '''
    with open(f'{pickle_name}.pickle', 'rb') as handle:
        events_per_midi = pickle.load(handle)
        events = {}
        files_containing_events = {}
        for midi_file in events_per_midi:
            for event in events_per_midi[midi_file]:
                if event not in events:
                    events[event] = 0
                    files_containing_events[event] = []
                events[event] += 1
                if midi_file not in files_containing_events[event]:
                    files_containing_events[event].append(midi_file)
        for event in sorted(events, key=lambda x: events[x]):
            print(f'{event}: {events[event]}')
            print(f'   {files_containing_events[event]}')


def get_midi_from_name(midi_file, pickle_name):
    '''
    IGNORE Testing function: Get a pickled dict and print the content of a particular song
    '''
    with open(f'{pickle_name}.pickle', 'rb') as handle:
        events_per_midi = pickle.load(handle)
        print(events_per_midi[midi_file])
