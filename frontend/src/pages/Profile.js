import {
  View,
  Text,
  Image,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
} from 'react-native';
import React from 'react';
import {useNavigation} from '@react-navigation/native';
import {getAuth, signOut} from '@react-native-firebase/auth';
// import { ScrollView } from 'react-native-gesture-handler';
const auth = getAuth();
const Profile = ({route}) => {
  const signOut = () => {
    console.log('Signout called');
    // Your authentication sign-out logic (e.g., Firebase.auth().signOut())
    // After signing out, navigate to the Login screen.
    navigation.navigate('Login');
  };

  const navigation = useNavigation();
  const {userD} = route.params;
  return (
    <View style={styles.container}>
      <ScrollView>
        <View style={{alignItems: 'center'}}>
          <Image
            source={require('../assets/house.jpg')}
            style={{width: 140, height: 140, borderRadius: 100, marginTop: 50}}
          />
        </View>
        <View style={styles.box}>
          <Image
            source={require('../assets/person.png')}
            style={styles.boxImage}
          />
          <Text style={styles.heading}>{userD.name}</Text>
        </View>

        <View style={styles.box}>
          <Image
            source={require('../assets/email.png')}
            style={styles.boxImage}
          />
          <Text style={styles.heading}>{userD.email}</Text>
        </View>
        <View style={styles.box}>
          <Image
            source={require('../assets/phone.png')}
            style={styles.boxImage}
          />
          <Text style={styles.heading}>{userD.phone}</Text>
        </View>
        <View style={styles.box}>
          <Image
            source={require('../assets/location.png')}
            style={styles.boxImage}
          />
          <Text style={styles.heading}>{userD.place}</Text>
        </View>

        <TouchableOpacity
          style={[styles.box, {backgroundColor: '#ff8717'}]}
          onPress={signOut}>
          <Text style={{color: '#FFFFFF', fontSize: 20}}>Logout</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};
const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  heading: {
    fontSize: 20,
    fontWeight: 'bold',
    marginLeft: 10,
    color: 'black',
  },
  box: {
    alignSelf: 'center',
    flexDirection: 'row',
    justifyContent: 'center',
    backgroundColor: '#fff',
    width: '90%',
    padding: 20,
    paddingBottom: 22,
    borderRadius: 10,
    shadowOpacity: 80,
    elevation: 15,
    marginTop: 40,
    // alignItems:'center'
  },
  boxImage: {
    width: 20,
    height: 20,
    marginTop: 5,
  },
});

export default Profile;
