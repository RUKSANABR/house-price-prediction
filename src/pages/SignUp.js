import {
  View,
  Text,
  SafeAreaView,
  StyleSheet,
  TextInput,
  TouchableOpacity,
} from 'react-native';
import React, {useState} from 'react';

import {Avatar} from '@rneui/base';

export default function Login({navigation}) {
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();
  const [name, setName] = useState();
  const onSubmit = () => {};

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.screen}>
        <Text style={styles.font}>Sign Up</Text>
      </View>
      <View style={styles.form1}>
        <Avatar source={require('../assets/person.png')} containerStyle={{marginLeft:5}}/>
        <TextInput
          placeholder="Enter Name"
          on
          onChangeText={setName}></TextInput>
      </View>
      <View style={[styles.form1, {marginTop: 20}]}>
        <Avatar
          size={32}
          source={require('../assets/email.png')}
          containerStyle={{marginLeft: 10, marginTop: 5}}
        />
        <TextInput
          placeholder="Enter Email"
          
          onChangeText={setEmail}></TextInput>
      </View>
      <View style={[styles.form1, {marginTop: 20}]}>
        <Avatar source={require('../assets/password.png')} containerStyle={{marginLeft:5}}/>
        <TextInput placeholder="Enter Password" onChangeText={setPassword} secureTextEntry={true} />
      </View>
      <View style={[styles.Button, {marginTop: 20}]}>
        <TouchableOpacity onPress={onSubmit}>
          <Text style={styles.buttonText}>SignUp</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.signup}>
        <Text>Already have an account?</Text>
        <Text
          style={styles.textColor}
          onPress={() => navigation.navigate('Login')}>
          {' '}
          Login
        </Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  screen: {
    alignItems: 'center',
  },
  font: {
    fontSize: 28,
    fontWeight: '500',
    color: '#333',
    marginBottom: 30,
  },
  form1: {
    flexDirection: 'row',
    borderColor: '#ccc',
    alignItems: 'center',
    borderWidth: 1,
    marginBottom: 2,
    width: '95%',
    borderRadius: 10,
  },
  Button: {
    width: '95%',
    height: 50,
    backgroundColor: '#E01E14',
    borderRadius: 10,
  },
  buttonText: {
    textAlign: 'center',
    justifyContent: 'center',
    fontSize: 20,
    marginTop: 10,
    color: 'white',
  },
  textColor: {
    color: '#EB5E52',
  },
  signup: {
    textAlign: 'center',
    flexDirection: 'row',
    marginTop: 50,
  },
});
