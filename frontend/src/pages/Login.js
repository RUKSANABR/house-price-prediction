import React, { useState } from 'react';
import {
  View,
  Text,
  SafeAreaView,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Alert,
  Image,
} from 'react-native';
import { Avatar } from '@rneui/base';
import { getAuth, signInWithEmailAndPassword } from '@react-native-firebase/auth';
import firestore from '@react-native-firebase/firestore';


const auth = getAuth(); // Get the authentication object

export default function Login({ navigation }) {
  const [email, setEmail] = useState();
  const [password, setPassword] = useState();
  const [showPassword, setShowPassword] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      // Check if any required field is empty
      Alert.alert('Invalied Login');
      return; // Exit the function
    }
  
  try {
    await signInWithEmailAndPassword(auth, email, password);
    const user = auth.currentUser;

    // Rest of your code...
    firestore()
      .collection('users')
      .where('email', '==', user.email)
      .get()
      .then((querySnapshot) => {
        if (!querySnapshot.empty) {
          const userD = querySnapshot.docs[0].data();

          console.log('User Type:', userD.userType);

          if (userD.userType === 'user') {
            navigation.reset({
              index: 0,
              routes: [{ name: 'Tabs', params: { userD } }],
            });
          } else {
            navigation.reset({
              index: 0,
              routes: [{ name: 'ContracterPage', params: { userD } }],
            });
          }
        }
      });
  } catch (error) {
    // console.error('Login Error:', error);
    Alert.alert('Invalied email or password'); // Move the Alert here
  }
};


  const togglePasswordVisibility = () => {
    setShowPassword((prevState) => !prevState);
  };
  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.screen}>
        <Text style={styles.font}>Login</Text>
      </View>
      <View style={styles.form1}>
        <Avatar
          size={32}
          source={require('../assets/email.png')}
          containerStyle={{marginLeft: 10, marginTop: 5}}
        />
       <TextInput
          placeholder="Enter Email"
          onChangeText={setEmail}
          style={{ color: 'black' }} placeholderTextColor="black"></TextInput>
      </View>
      <View style={[styles.form1, {marginTop: 20}]}>
        <Avatar
          source={require('../assets/password.png')}
          containerStyle={{marginLeft: 5}}
        />
        <TextInput placeholder="Enter Password" onChangeText={setPassword}  secureTextEntry={!showPassword} style={{ color: 'black' }} placeholderTextColor="black"/>
        <TouchableOpacity
              style={{
                position: "absolute",
                top: 10,
                right: 30,
              }}
              onPress={togglePasswordVisibility}
            >
              <Image
                source={
                  showPassword
                    ? require("../assets/eye.png")
                    : require("../assets/blind.png")
                }
                style={{
                  width: 30,
                  height: 30,
                  resizeMode: "contain",
                }}
              />
            </TouchableOpacity>
          {/*  */}
          </View>
      <View style={[styles.Button, {marginTop: 20}]}>
        <TouchableOpacity onPress={handleLogin}>
          <Text style={styles.buttonText}>Login</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.signup}>
        <Text style={{color:"black"}}>Don't have an account?</Text>
        <Text
          style={styles.textColor}
          onPress={() => navigation.navigate('SignUp')}>
          {' '}
          Create
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
